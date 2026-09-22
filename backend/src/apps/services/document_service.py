"""文档登记与生命周期：上传 → ingest → ready/failed，及详情/源文件/删除/重试。

P0 闭环约定：
- doc_id 使用 UUID，避免 doc_{count+1} 在并发/删除后的撞号
- 登记即保存 SHA256、size、created_at；ingest 成败回写 counts 与 error
- 状态机统一为 uploaded/indexing/ready/failed；同文档并发 ingest 返回 409
- 上传大小受 upload.max_bytes（默认 20 MiB）限制
- 删除按 状态记录 → ES 索引 → 源文件 顺序清理；ES 失败保留记录可重试
数据归属过滤（owner 校验）归 P1；当前端点只要求登录。
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.src.apps.services.common_service import ServiceError, ensure_upload_dir
from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.apps.services.state_store import delete_document, get_document, list_documents, upsert_document
from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.config.settings import settings

_ALLOWED_EXT = {"pdf", "md", "txt", "html", "htm"}
_CONTENT_TYPE_TO_EXT = {
    "application/pdf": "pdf",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/plain": "txt",
    "text/html": "html",
}
_ERROR_MAX_CHARS = 500


def _upload_max_bytes() -> int:
    return settings.integer('upload.max_bytes', positive=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def public_document(doc: dict) -> dict:
    """对外视图：隐藏服务器本地路径，其余字段（含 sha256/counts）原样返回。"""
    return {key: value for key, value in doc.items() if key != "file_path"}


class DocumentService:
    def __init__(self) -> None:
        # Parsing binds embedding/metadata providers. Keep it out of API import
        # and startup so uploads and health checks do not initialize heavy models.
        self._pipeline: IngestionPipeline | None = None
        # 同文档并发 ingest 的进程内保护：ingest 是同步重任务（线程池执行），
        # 仅靠状态字段检查存在 TOCTOU 窗口
        self._indexing: set[str] = set()
        self._indexing_guard = threading.Lock()

    @property
    def pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = IngestionPipeline()
        return self._pipeline

    def _resolve_ext(self, file_name: str, content_type: str | None) -> str:
        by_name = Path(file_name).suffix.lower().lstrip(".")
        if by_name in _ALLOWED_EXT:
            return by_name
        if content_type:
            by_type = _CONTENT_TYPE_TO_EXT.get(content_type.lower())
            if by_type in _ALLOWED_EXT:
                return by_type
        raise ServiceError(
            "INVALID_FILE_TYPE",
            f"不支持的文件类型: {file_name}",
            {"allowed": sorted(_ALLOWED_EXT)},
            status_code=422,
        )

    def upload_file(
        self,
        file_name: str,
        content_type: str | None,
        content: bytes,
        *,
        owner_id: str | None = None,
    ) -> dict:
        clean_name = (file_name or "").strip()
        if not clean_name:
            raise ServiceError("INVALID_FILE_NAME", "file_name 不能为空")
        if not content:
            raise ServiceError("EMPTY_FILE", "上传文件不能为空")
        max_bytes = _upload_max_bytes()
        if len(content) > max_bytes:
            raise ServiceError(
                "FILE_TOO_LARGE",
                f"上传文件超过大小限制 {max_bytes} 字节",
                {"size": len(content), "max_bytes": max_bytes},
                status_code=413,
            )

        file_type = self._resolve_ext(clean_name, content_type)
        doc_id = uuid.uuid4().hex

        upload_dir = ensure_upload_dir()
        suffix = f".{file_type}"
        stored_name = f"{doc_id}{suffix}"
        file_path = upload_dir / stored_name
        file_path.write_bytes(content)

        doc = {
            "doc_id": doc_id,
            "name": clean_name,
            "file_type": file_type,
            "file_path": str(file_path),
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "status": "uploaded",
            "created_at": _now_iso(),
            "error": None,
            "counts": None,
            # 归属权威字段：资产读取（P0-A）按它做 JOIN 鉴权；过滤归 P1
            "owner_id": owner_id,
        }
        upsert_document(doc)
        return public_document(doc)

    def ingest(self, doc_id: str) -> dict:
        if not doc_id.strip():
            raise ServiceError("INVALID_DOC_ID", "doc_id 不能为空", status_code=422)

        doc = get_document(doc_id)
        if not doc:
            raise ServiceError("DOC_NOT_FOUND", f"doc_id 不存在: {doc_id}", status_code=404)

        with self._indexing_guard:
            if doc_id in self._indexing:
                raise ServiceError(
                    "DOC_INDEXING_IN_PROGRESS",
                    f"文档正在索引中，请稍后再试: {doc_id}",
                    status_code=409,
                )
            self._indexing.add(doc_id)
        try:
            return self._run_ingest(doc)
        finally:
            with self._indexing_guard:
                self._indexing.discard(doc_id)

    def _run_ingest(self, doc: dict) -> dict:
        file_path = Path(doc.get("file_path", ""))
        if not file_path.exists():
            raise ServiceError("FILE_NOT_FOUND", "源文件不存在", {"file_path": str(file_path)}, status_code=404)

        doc["status"] = "indexing"
        doc["error"] = None
        upsert_document(doc)
        try:
            result = self.pipeline.run(
                doc["doc_id"],
                parse_config={
                    "file_type": doc["file_type"],
                    "file_path": str(file_path),
                    "doc_name": doc["name"],
                },
                chunk_config=ChunkConfig(),
                owner_id=doc.get("owner_id"),
            )
        except Exception as exc:
            doc["status"] = "failed"
            doc["error"] = str(exc)[:_ERROR_MAX_CHARS]
            upsert_document(doc)
            raise ServiceError(
                "INGEST_FAILED", "ingest 失败", {"reason": str(exc)[:_ERROR_MAX_CHARS]}, status_code=502,
            ) from exc
        doc["status"] = "ready"
        doc["counts"] = {
            "parsed": result.get("parsed_count"),
            "chunk": result.get("chunk_count"),
            "indexed": result.get("indexed_count"),
        }
        upsert_document(doc)
        return result

    def detail(self, doc_id: str) -> dict:
        doc = get_document(doc_id)
        if not doc:
            raise ServiceError("DOC_NOT_FOUND", f"doc_id 不存在: {doc_id}", status_code=404)
        return public_document(doc)

    def source_path(self, doc_id: str) -> Path:
        """返回源文件真实路径；解析结果必须仍在 uploads 目录内，防止记录被篡改后外泄文件。"""
        doc = get_document(doc_id)
        if not doc:
            raise ServiceError("DOC_NOT_FOUND", f"doc_id 不存在: {doc_id}", status_code=404)
        file_path = Path(doc.get("file_path", ""))
        upload_root = ensure_upload_dir().resolve()
        resolved = file_path.resolve()
        if not resolved.is_relative_to(upload_root):
            raise ServiceError("FILE_NOT_FOUND", "源文件不存在", {"doc_id": doc_id}, status_code=404)
        if not resolved.exists():
            raise ServiceError("FILE_NOT_FOUND", "源文件不存在", {"doc_id": doc_id}, status_code=404)
        return resolved

    def delete(self, doc_id: str) -> dict:
        doc = get_document(doc_id)
        if not doc:
            raise ServiceError("DOC_NOT_FOUND", f"doc_id 不存在: {doc_id}", status_code=404)
        if doc.get("status") == "indexing" or doc_id in self._indexing:
            raise ServiceError(
                "DOC_INDEXING_IN_PROGRESS",
                f"文档正在索引中，暂不能删除: {doc_id}",
                status_code=409,
            )
        # 先清 ES：失败时保留本地记录，调用方可直接重试，避免留下无索引的孤儿文档
        try:
            self.pipeline.embedding_indexer.delete_document(doc_id)
        except Exception as exc:
            raise ServiceError(
                "DELETE_FAILED", "清理索引失败，请重试", {"reason": str(exc)[:_ERROR_MAX_CHARS]}, status_code=502,
            ) from exc
        Path(doc.get("file_path", "")).unlink(missing_ok=True)
        delete_document(doc_id)
        return {"doc_id": doc_id, "deleted": True}

    def list_documents(self) -> list[dict]:
        return [public_document(doc) for doc in list_documents()]
