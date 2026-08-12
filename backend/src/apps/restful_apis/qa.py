from typing import Any, Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.qa_service import QAService

router = APIRouter(tags=["qa"])
service = QAService()


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16_000)


class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)
    retrieval_config: Optional[dict[str, Any]] = Field(default=None)


@router.post("/qa/query")
async def query_qa(req: QueryRequest, request: Request) -> dict:
    require_authenticated(request)
    history = [{"role": item.role, "content": item.content} for item in req.history]
    data = service.query(
        question=req.question,
        history=history,
        retrieval_config=req.retrieval_config,
    )
    return ok(data)
