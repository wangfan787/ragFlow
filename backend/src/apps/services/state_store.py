from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from copy import deepcopy

from backend.src.config.data_paths import database_path
# from backend.src.adapters.vector_store.memory_vector_store import MemoryVectorStore
# from backend.src.apps.services.vector_state_store import clear_vectors


# 文档元数据持久化到 SQLite（与向量索引使用同一个数据库文件），
# 以确保服务重启后文档注册信息不会丢失。
# 文档切片（Chunks）仅保存在内存中：服务重启后，
# 关键词检索通道将回退使用存储在向量旁边的 payload 数据。
_DOC_CHUNKS: dict[str, list[dict]] = {}
# 切片版本号，每次切片发生变更时递增，用作派生关键词索引的缓存键
_CHUNKS_VERSION = 0


def _connect() -> sqlite3.Connection:
    """
    创建并返回一个 SQLite 数据库连接。
    如果数据库文件或 documents 表不存在，则自动创建。
    """
    path = database_path()
    # 确保数据库所在目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # 初始化文档表：doc_id 为主键，payload 存储 JSON 序列化的文档内容
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """
    )
    return conn


def upsert_document(doc: dict) -> None:
    """
    插入或更新文档。
    如果 doc_id 已存在则更新 payload，否则插入新记录。
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, payload)
            VALUES (?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET payload = excluded.payload
            """,
            (str(doc["doc_id"]), json.dumps(doc, ensure_ascii=False)),
        )


def get_document(doc_id: str) -> dict | None:
    """
    根据 doc_id 获取单个文档。
    如果文档不存在则返回 None。
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload FROM documents WHERE doc_id = ?", (str(doc_id),)
        ).fetchone()
    return json.loads(row[0]) if row else None


def list_documents() -> list[dict]:
    """
    列出所有已持久化的文档，按插入顺序（rowid）排序。
    """
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM documents ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


def chunks_version() -> int:
    """
    返回当前切片版本号。
    每当切片发生增删改操作时版本号递增，可作为派生关键词索引的缓存失效依据。
    """
    return _CHUNKS_VERSION


def save_chunks(doc_id: str, chunks: Iterable[dict]) -> None:
    """
    将指定文档的切片保存到内存缓存中。
    使用深拷贝防止外部引用修改缓存数据，同时递增版本号以触发缓存刷新。
    """
    global _CHUNKS_VERSION
    _DOC_CHUNKS[doc_id] = [deepcopy(chunk) for chunk in chunks]
    _CHUNKS_VERSION += 1


def list_chunks(doc_id: str) -> list[dict]:
    """
    获取指定文档的所有切片（深拷贝副本）。
    如果该文档没有切片则返回空列表。
    """
    return deepcopy(_DOC_CHUNKS.get(doc_id, []))


def list_all_chunks() -> list[dict]:
    """
    获取内存中所有文档的全部切片（深拷贝副本）。
    """
    all_chunks: list[dict] = []
    for chunks in _DOC_CHUNKS.values():
        all_chunks.extend(deepcopy(chunks))
    return all_chunks


def reset_all() -> None:
    """
    重置整个文档存储系统：
    1. 清空 SQLite 中的文档表
    2. 清空内存中的切片缓存
    3. 递增版本号使缓存失效
    4. 清除向量索引及全局向量存储
    """
    global _CHUNKS_VERSION
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
    _DOC_CHUNKS.clear()
    _CHUNKS_VERSION += 1
    # clear_vectors()
    # MemoryVectorStore.clear_global()
