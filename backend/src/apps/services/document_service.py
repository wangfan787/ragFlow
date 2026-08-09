from __future__ import annotations

from pathlib import Path

from backend.src.apps.services.common_service import ServiceError, ensure_upload_dir
from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.apps.services.state_store import get_document, list_documents, upsert_document
from backend.src.chunking.chunk_config import ChunkConfig

_ALLOWED_EXT = {"pdf", "md"}
_CONTENT_TYPE_TO_EXT = {
    "application/pdf": "pdf",
    "text/markdown": "md",
    "text/x-markdown": "md",
}


class DocumentService:
    def __init__(self) -> None:
        # Parsing binds embedding/metadata providers. Keep it out of API import
        # and startup so uploads and health checks do not initialize heavy models.
        self._pipeline: IngestionPipeline | None = None

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
        )

    def upload_file(self, file_name: str, content_type: str | None, content: bytes) -> dict:
        clean_name = (file_name or "").strip()
        if not clean_name:
            raise ServiceError("INVALID_FILE_NAME", "file_name 不能为空")
        if not content:
            raise ServiceError("EMPTY_FILE", "上传文件不能为空")

        file_type = self._resolve_ext(clean_name, content_type)
        doc_id = f"doc_{len(list_documents()) + 1}"

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
            "status": "PENDING",
        }
        upsert_document(doc)
        return doc

    def ingest(self, doc_id: str) -> dict:
        if not doc_id.strip():
            raise ServiceError("INVALID_DOC_ID", "doc_id 不能为空")

        doc = get_document(doc_id)
        if not doc:
            raise ServiceError("DOC_NOT_FOUND", f"doc_id 不存在: {doc_id}")

        file_path = Path(doc.get("file_path", ""))
        if not file_path.exists():
            raise ServiceError("FILE_NOT_FOUND", "源文件不存在", {"file_path": str(file_path)})

        doc["status"] = "PROCESSING"
        upsert_document(doc)
        try:
            result = self.pipeline.run(
                doc_id,
                parse_config={
                    "file_type": doc["file_type"],
                    "file_path": str(file_path),
                    "doc_name": doc["name"],
                },
                chunk_config=ChunkConfig(),
            )
            doc["status"] = "SUCCESS"
            upsert_document(doc)
            return result
        except Exception as exc:  # pragma: no cover
            doc["status"] = "FAILED"
            upsert_document(doc)
            raise ServiceError("INGEST_FAILED", "ingest 失败", {"reason": str(exc)}) from exc

    def list_documents(self) -> list[dict]:
        return list_documents()