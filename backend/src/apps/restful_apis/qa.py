from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.qa_service import QAService

router = APIRouter(tags=["qa"])
service = QAService()


class QueryRequest(BaseModel):
    """问答请求；retrieval_config / qa_config 支持逐请求覆盖（评测 Runner 依赖）。"""

    question: str
    # 可选的请求级覆盖，结构与 RetrievalConfig / QAConfig 字段一致；
    # 不传时使用服务默认配置。非法字段/取值返回 422，而不是静默忽略。
    retrieval_config: dict[str, Any] | None = None
    qa_config: dict[str, Any] | None = None


@router.post("/qa/query")
async def query_qa(req: QueryRequest, request: Request) -> dict:
    require_authenticated(request)
    try:
        data = service.query(
            question=req.question,
            retrieval_config=req.retrieval_config,
            qa_config=req.qa_config,
        )
    except ValueError as exc:
        # 配置构建阶段的校验错误（未知字段、越界取值）属于请求错误
        raise HTTPException(422, f"检索或 QA 配置不合法：{exc}") from exc
    return ok(data)
