"""图片资产：SHA256 去重保存 + SQLite 归属登记 + 受控读取。

P0-A 安全底线：
1. 调用方只能提交 document_id/asset_id，不能提交服务器本地路径；
2. 读取前必须用同一条 SQL 同时校验资源 ID、文档归属与当前用户；
3. storage_key 由后端生成；读取前确认规范化路径仍在资产根目录内，
   拒绝 `..` 等路径穿越；
4. SHA256 只作为查重标识，不是权限凭证。
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.src.config.data_paths import data_dir, database_path


class AssetPathError(ValueError):
    """storage_key 非法或试图逃出资产根目录。"""


class AssetNotFoundError(LookupError):
    """资产记录存在但文件缺失。"""


class AssetFileStore:
    """原图以 sha256 摘要为名落盘；同内容幂等，不重复写。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or data_dir() / "assets").resolve()

    def _resolve(self, storage_key: str) -> Path:
        if not storage_key or not isinstance(storage_key, str):
            raise AssetPathError("storage_key 不能为空")
        if "\\" in storage_key or storage_key.startswith("/"):
            raise AssetPathError(f"storage_key 非法: {storage_key!r}")
        candidate = (self.root / storage_key).resolve()
        if self.root not in candidate.parents:
            raise AssetPathError(f"storage_key 逃出资产根目录: {storage_key!r}")
        return candidate

    def save(self, content: bytes, ext: str) -> tuple[str, str]:
        """保存原图，返回 (asset_id, storage_key)。同内容重复保存幂等。"""
        digest = hashlib.sha256(content).hexdigest()
        asset_id = f"sha256:{digest}"
        storage_key = f"{digest[:2]}/{digest}{ext}"
        path = self._resolve(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return asset_id, storage_key

    def read(self, storage_key: str) -> bytes:
        path = self._resolve(storage_key)
        if not path.is_file():
            raise AssetNotFoundError(f"资产文件缺失: {storage_key}")
        return path.read_bytes()


class AssetRegistry:
    """资产归属登记；与 documents 表同库，归属校验用一条 JOIN 完成。"""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            storage_key TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            owner_id TEXT,
            description_status TEXT NOT NULL DEFAULT 'pending',
            vlm_model TEXT,
            vlm_prompt_version TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (asset_id, document_id)
        )
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        path = self._db_path or database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute(self._SCHEMA)
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {key: row[key] for key in row.keys()}

    def register(
        self,
        *,
        asset_id: str,
        document_id: str,
        storage_key: str,
        mime_type: str,
        sha256: str,
        size_bytes: int,
        owner_id: str | None,
    ) -> dict:
        """登记资产归属；同一 (asset_id, document_id) 重复登记幂等。

        同一张图片被多个文档引用时各写一行，归属随文档分别判定。
        """
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (
                    asset_id, document_id, storage_key, mime_type, sha256,
                    size_bytes, owner_id, description_status, vlm_prompt_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?)
                ON CONFLICT(asset_id, document_id) DO NOTHING
                """,
                (asset_id, document_id, storage_key, mime_type, sha256,
                 size_bytes, owner_id, created_at),
            )
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ? AND document_id = ?",
                (asset_id, document_id),
            ).fetchone()
        assert row is not None
        return self._row_to_dict(row)

    def mark_description(
        self,
        *,
        asset_id: str,
        document_id: str,
        status: str,
        vlm_model: str | None = None,
        vlm_prompt_version: str | None = None,
    ) -> None:
        """回写 VLM 描述结果状态；failed/skipped 也如实记录，不删资产。"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE assets
                SET description_status = ?, vlm_model = ?, vlm_prompt_version = ?
                WHERE asset_id = ? AND document_id = ?
                """,
                (status, vlm_model, vlm_prompt_version, asset_id, document_id),
            )

    def get_asset(self, asset_id: str, document_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ? AND document_id = ?",
                (asset_id, document_id),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def authorized_asset(
        self, asset_id: str, document_id: str, owner_id: str | None,
    ) -> dict | None:
        """归属校验：资源 ID + 文档 ID + 当前用户必须在同一条查询里满足。

        documents.payload.owner_id 是归属权威；查不到一律返回 None，
        调用方不得先按 ID 查出再补鉴权。
        """
        if not owner_id:
            return None
        query = """
            SELECT a.*
            FROM assets a
            JOIN documents d ON d.doc_id = a.document_id
            WHERE a.asset_id = ?
              AND a.document_id = ?
              AND json_extract(d.payload, '$.owner_id') = ?
        """
        with self._connect() as conn:
            row = conn.execute(query, (asset_id, document_id, owner_id)).fetchone()
        return self._row_to_dict(row) if row else None
