"""SQLite 会话存档。模型调用在事务外，所有写入同时检查版本和请求租约。"""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from backend.src.apps.services.common_service import ServiceError
from backend.src.config.data_paths import data_dir
from backend.src.config.settings import settings


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def conflict(message="会话已被其他请求修改，请重新加载"):
    return ServiceError("SESSION_CONFLICT", message, status_code=409)


@dataclass
class SessionRequest:
    session: dict
    request_id: str
    token: str
    cached: dict | None = None


class SessionStore:
    def __init__(self, path: Path | None = None):
        # 路径延迟求值；导入/健康检查不创建文件或连接外部服务。
        self.path = path

    @contextmanager
    def _db(self):
        path = self.path or data_dir() / "sessions.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            pages = settings.integer("sessions.max_storage_bytes", positive=True) // db.execute("PRAGMA page_size").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={max(1, pages)}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, version INTEGER NOT NULL,
                    state TEXT NOT NULL, updated REAL NOT NULL, expires REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_expiry ON sessions(expires);
                CREATE TABLE IF NOT EXISTS requests (
                    owner TEXT NOT NULL, id TEXT NOT NULL, session_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, token TEXT NOT NULL,
                    status TEXT NOT NULL, updated REAL NOT NULL, result TEXT,
                    PRIMARY KEY(owner,id),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS request_session ON requests(session_id);
            """)
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.OperationalError as exc:
            db.rollback()
            if "full" in str(exc).lower():
                raise ServiceError("SESSION_CAPACITY", "会话存储已满", status_code=413) from exc
            raise
        finally:
            db.close()

    @staticmethod
    def _load(db, owner, session_id):
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ServiceError("SESSION_NOT_FOUND", "会话不存在", status_code=404)
        if row["owner"] != owner:
            raise ServiceError("SESSION_FORBIDDEN", "无权访问该会话", status_code=403)
        if row["expires"] <= time.time():
            raise ServiceError("SESSION_EXPIRED", "会话已过期", status_code=410)
        return {"session_id": row["id"], "owner_id": row["owner"], "version": row["version"],
                "updated_at": row["updated"], "expires_at": row["expires"], **json.loads(row["state"])}

    @staticmethod
    def _create(db, owner):
        db.execute("DELETE FROM sessions WHERE expires<=?", (time.time(),))
        if db.execute("SELECT count(*) FROM sessions").fetchone()[0] >= settings.integer("sessions.max_sessions", positive=True):
            raise ServiceError("SESSION_CAPACITY", "会话数量已达上限", status_code=413)
        session_id = str(uuid.uuid4())
        now = time.time()
        state = {"turns": [], "summary": "", "summarized_through": 0, "compaction_attempt": None}
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)", (
            session_id, owner, 0, encode(state), now, now + settings.integer("sessions.ttl_seconds", positive=True)))
        return session_id

    def create(self, owner: str) -> dict:
        with self._db() as db:
            return self._load(db, owner, self._create(db, owner))

    def read(self, owner: str, session_id: str) -> dict:
        with self._db() as db:
            return self._load(db, owner, session_id)

    def begin(self, owner: str, session_id: str | None, request_id: str | None, content: dict) -> SessionRequest:
        request_id = request_id or str(uuid.uuid4())
        if not owner or not request_id.strip() or len(request_id) > 128:
            raise ServiceError("INVALID_REQUEST_ID", "请求标识不能为空且最长 128 字符", status_code=422)
        fingerprint = hashlib.sha256(encode(content).encode()).hexdigest()
        now, token = time.time(), str(uuid.uuid4())
        with self._db() as db:
            row = db.execute("SELECT * FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if row:
                if row["fingerprint"] != fingerprint or (session_id and session_id != row["session_id"]):
                    raise conflict("同一 request_id 不能用于不同请求")
                session = self._load(db, owner, row["session_id"])
                if row["status"] == "done":
                    return SessionRequest(session, request_id, row["token"], json.loads(row["result"]))
                if row["status"] == "pending" and now - row["updated"] < settings.integer("sessions.request_lease_seconds", positive=True):
                    raise conflict("该 request_id 正在执行")
                db.execute("UPDATE requests SET token=?, status='pending', updated=? WHERE owner=? AND id=?", (token, now, owner, request_id))
            else:
                session = self._load(db, owner, session_id) if session_id else self._load(db, owner, self._create(db, owner))
                db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
                if db.execute("SELECT count(*) FROM requests WHERE session_id=?", (session["session_id"],)).fetchone()[0] >= settings.integer("sessions.max_requests", positive=True):
                    raise ServiceError("SESSION_CAPACITY", "会话请求数已达上限，请创建新会话", status_code=413)
                db.execute("INSERT INTO requests VALUES(?,?,?,?,?,'pending',?,NULL)", (owner, request_id, session["session_id"], fingerprint, token, now))
            return SessionRequest(session, request_id, token)

    def _check(self, db, request):
        session = request.session
        current = self._load(db, session["owner_id"], session["session_id"])
        row = db.execute("SELECT token,status FROM requests WHERE owner=? AND id=?", (session["owner_id"], request.request_id)).fetchone()
        if current["version"] != session["version"] or not row or row["token"] != request.token or row["status"] != "pending":
            raise conflict()

    def _save(self, db, request, state):
        session = request.session
        encoded = encode(state)
        result_bytes = db.execute("SELECT coalesce(sum(length(CAST(result AS BLOB))),0) FROM requests WHERE session_id=?", (session["session_id"],)).fetchone()[0]
        if len(encoded.encode()) + result_bytes > settings.integer("sessions.max_session_bytes", positive=True) or len(state["turns"]) > settings.integer("sessions.max_turns", positive=True):
            raise ServiceError("SESSION_CAPACITY", "会话容量已达上限，请创建新会话", status_code=413)
        now = time.time()
        db.execute("UPDATE sessions SET state=?, version=version+1, updated=?, expires=? WHERE id=?", (
            encoded, now, now + settings.integer("sessions.ttl_seconds", positive=True), session["session_id"]))
        # 更新租约避免分段压缩中途被正常重试接管。
        db.execute("UPDATE requests SET updated=? WHERE owner=? AND id=?", (now, session["owner_id"], request.request_id))
        return self._load(db, session["owner_id"], session["session_id"])

    @staticmethod
    def _state(session):
        return {key: session[key] for key in ("turns", "summary", "summarized_through", "compaction_attempt")}

    def compact(self, request: SessionRequest, summary: str, through: int, attempt: dict):
        with self._db() as db:
            self._check(db, request)
            state = self._state(request.session)
            if through < state["summarized_through"] or through > len(state["turns"]):
                raise ValueError("invalid summary coverage")
            if through > state["summarized_through"] and not summary.strip():
                raise ValueError("summary cannot be empty")
            state.update(summary=summary, summarized_through=through, compaction_attempt=attempt)
            saved = self._save(db, request, state)
        request.session = saved

    def complete(self, request: SessionRequest, question: str, payload: dict) -> dict:
        with self._db() as db:
            self._check(db, request)
            session = request.session
            result = {**payload, "session_id": session["session_id"], "request_id": request.request_id,
                      "session_version": session["version"] + 1}
            state = self._state(session)
            state["turns"] = [*state["turns"], {"seq": len(state["turns"]) + 1, "question": question,
                "answer": result["answer"], "status": result["status"], "citations": result["citations"]}]
            db.execute("UPDATE requests SET status='done',result=? WHERE owner=? AND id=?", (encode(result), session["owner_id"], request.request_id))
            saved = self._save(db, request, state)
        request.session = saved
        request.cached = result
        return result

    def abandon(self, request: SessionRequest):
        with self._db() as db:
            db.execute("UPDATE requests SET status='failed' WHERE owner=? AND id=? AND token=? AND status='pending'", (request.session["owner_id"], request.request_id, request.token))
