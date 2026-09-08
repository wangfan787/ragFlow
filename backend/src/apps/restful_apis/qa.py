from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.qa_service import QAService

router = APIRouter(tags=["qa"])
service = QAService()


class QueryRequest(BaseModel):
    question: str


@router.post("/qa/query")
async def query_qa(req: QueryRequest, request: Request) -> dict:
    require_authenticated(request)
    data = service.query(question=req.question)
    return ok(data)
