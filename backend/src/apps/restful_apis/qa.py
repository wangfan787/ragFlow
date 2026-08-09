from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.qa_service import QAService

router = APIRouter(tags=["qa"])
service = QAService()


class QueryRequest(BaseModel):
    question: str
    retrieval_config: Optional[dict[str, Any]] = Field(default=None)


@router.post("/qa/query")
async def query_qa(req: QueryRequest, request: Request) -> dict:
    require_authenticated(request)
    data = service.query(question=req.question, retrieval_config=req.retrieval_config)
    return ok(data)