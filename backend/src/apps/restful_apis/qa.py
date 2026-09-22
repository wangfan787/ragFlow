"""QA API：同步问答与 SSE 流式问答。

线程模型：/qa/query 是同步重任务（检索+生成可到秒级），用普通 def 端点
交给 FastAPI 线程池，不阻塞事件循环；/qa/query/stream 用异步生成器编码
SSE，生成器内的重活经 asyncio.to_thread 下放，客户端断开时停止发送。
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.apps.services.common_service import ok, require_authenticated
from backend.src.apps.services.qa_service import QAService
from backend.src.config.qa_config import build_qa_config
from backend.src.config.retrieval_config import build_retrieval_config

router = APIRouter(tags=["qa"])
service = QAService()


class QueryRequest(BaseModel):
    """问答请求；retrieval_config / qa_config 支持逐请求覆盖（评测 Runner 依赖）。"""

    question: str
    # 可选多轮对话历史（[{role, content}, ...]，仅用于查询改写成独立问句，
    # 不进入生成提示词；形状由改写服务清洗，异常项忽略，无历史时改写整体跳过）
    history: list[dict[str, Any]] | None = None
    # 可选的请求级覆盖，结构与 RetrievalConfig / QAConfig 字段一致；
    # 不传时使用服务默认配置。非法字段/取值返回 422，而不是静默忽略。
    retrieval_config: dict[str, Any] | None = None
    qa_config: dict[str, Any] | None = None


def _build_configs(req: QueryRequest):
    """预构建请求级配置：把配置校验错误挡在流开始之前，才能以 422 返回。"""
    retrieval_config = build_retrieval_config(req.retrieval_config)
    qa_config = build_qa_config(req.qa_config) if req.qa_config is not None else None
    return retrieval_config, qa_config


@router.post("/qa/query")
def query_qa(req: QueryRequest, request: Request) -> dict:
    require_authenticated(request)
    try:
        retrieval_config, qa_config = _build_configs(req)
    except ValueError as exc:
        # 配置构建阶段的校验错误（未知字段、越界取值）属于请求错误
        raise HTTPException(422, f"检索或 QA 配置不合法：{exc}") from exc
    data = service.query(
        question=req.question,
        retrieval_config=retrieval_config,
        qa_config=qa_config,
        history=req.history,
    )
    return ok(data)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/qa/query/stream")
async def query_qa_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    require_authenticated(request)
    try:
        retrieval_config, qa_config = _build_configs(req)
    except ValueError as exc:
        raise HTTPException(422, f"检索或 QA 配置不合法：{exc}") from exc

    async def event_source():
        events = service.query_stream(
            question=req.question,
            retrieval_config=retrieval_config,
            qa_config=qa_config,
            history=req.history,
        )
        sentinel = object()
        try:
            while True:
                item = await asyncio.to_thread(next, events, sentinel)
                if item is sentinel or item is None:
                    break
                event, data = item
                if await request.is_disconnected():
                    break
                yield _sse(event, data)
        finally:
            # 客户端断开或正常结束都关闭同步生成器；取消是尽力而为：
            # 正在执行的模型分块完成后即停止
            await asyncio.to_thread(events.close)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
