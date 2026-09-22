"""文档生命周期 API：上传 / 列表 / 详情 / 源文件 / 删除 / 重试。

线程模型：同步重任务（ingest、删除、文件读取）用普通 def 端点，
由 FastAPI 放入线程池执行，不阻塞事件循环；上传端点需要 await 文件读取，
同步登记部分用 asyncio.to_thread 下放。
"""

import asyncio
import mimetypes

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.document_service import DocumentService


router = APIRouter(tags=["documents"])
service = DocumentService()


class IngestRequest(BaseModel):
    doc_id: str


@router.post("/documents")
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    claims = require_authenticated(request)
    payload = await file.read()
    data = await asyncio.to_thread(
        service.upload_file,
        file.filename or "",
        file.content_type,
        payload,
        owner_id=str(claims.get("sub") or "") or None,
    )
    return ok(data)


@router.get("/documents")
def list_documents(request: Request) -> dict:
    require_authenticated(request)
    return ok({"documents": service.list_documents()})


@router.get("/documents/{doc_id}")
def document_detail(doc_id: str, request: Request) -> dict:
    require_authenticated(request)
    return ok(service.detail(doc_id))


@router.get("/documents/{doc_id}/source")
def download_source(doc_id: str, request: Request) -> FileResponse:
    """返回上传的原始文件；路径边界与存在性校验在 service.source_path 内完成。"""
    require_authenticated(request)
    file_path = service.source_path(doc_id)
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=media_type)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, request: Request) -> dict:
    require_authenticated(request)
    return ok(service.delete(doc_id))


@router.post("/documents/ingest")
def ingest_document(req: IngestRequest, request: Request) -> dict:
    require_authenticated(request)
    data = service.ingest(doc_id=req.doc_id)
    return ok(data)


@router.post("/documents/{doc_id}/retry")
def retry_document(doc_id: str, request: Request) -> dict:
    """失败/完成后重新 ingest；与 ingest 共用同一状态机与 409 并发保护。"""
    require_authenticated(request)
    data = service.ingest(doc_id=doc_id)
    return ok(data)
