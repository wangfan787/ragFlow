from fastapi import APIRouter, File, UploadFile, Request
from pydantic import BaseModel

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.document_service import DocumentService


router = APIRouter(tags=["documents"])
service = DocumentService()

class IngestRequest(BaseModel):
    doc_id: str

@router.post("/documents")
async def upload_document(request: Request,file:UploadFile = File(...)) ->dict:
    require_authenticated(request)
    payload = await file.read()
    data = service.upload_file(file_name=file.filename or "",content_type=file.content_type,content=payload)
    return ok(data)

@router.get("/documents")
async def upload_document(request: Request) ->dict:
    require_authenticated(request)
    return ok({"documents":service.list_documents()})

@router.post("/documents/ingest")
async def upload_document(req: IngestRequest,request:Request) ->dict:
    require_authenticated(request)
    data = service.ingest(doc_id = req.doc_id)
    return ok(data)