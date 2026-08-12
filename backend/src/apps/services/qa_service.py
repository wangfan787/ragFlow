from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict

from backend.src.apps.services.common_service import ServiceError
from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.apps.services.query_rewrite import QueryRewriteService
from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.citation.citation_service import CitationService
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.config.settings import settings
from backend.src.contracts import ChatModel
from backend.src.infrastructure.openai_chat import OpenAIChat, thinking_extra_body
from backend.src.retrieval.hybrid_router import HybridRouter
from backend.src.retrieval.metadata_fields import retrieved_chunk_payload
from backend.src.retrieval.models import RetrievedChunk

logger = logging.getLogger("mvp_api")
AnswerLLM = Callable[[Sequence[dict[str, str]]], str]

_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请只依据用户提供的检索证据回答。"
    "需要综合多个证据，不要只摘抄第一条证据。"
    "如果证据不足或与问题无关，请回答“未检索到相关证据”。"
    "回答应完整、结构清晰、使用中文；必要时保留代码或表格要点。"
    "引用证据时使用 [1]、[2] 这样的编号。"
)


class _ConfiguredChatModel:
    """Capability wrapper for test callables and the no-LLM fallback path."""

    def __init__(self, callback: AnswerLLM | None) -> None:
        self.callback = callback
        self.model_name = "configured-callable"
        self.context_limit_tokens = settings.integer(
            "MVP_QA_LLM_CONTEXT_TOKENS", 32768, positive=True
        )
        self.completion_reserve_tokens = settings.integer(
            "MVP_QA_COMPLETION_RESERVE_TOKENS",
            1024,
            positive=True,
        )
        self._counter = SimpleTokenCounter()

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        return 2 + sum(
            4
            + self._counter.count(str(message.get("role", "")))
            + self._counter.count(str(message.get("content", "")))
            for message in messages
        )

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        if self.callback is None:
            raise RuntimeError("QA LLM is not configured")
        return self.callback(messages)


def build_answer_model_from_env() -> ChatModel | None:
    api_key = settings.first("MVP_QA_LLM_API_KEY", "MVP_METADATA_LLM_API_KEY")
    model = settings.first("MVP_QA_LLM_MODEL", "MVP_METADATA_LLM_MODEL")
    base_url = settings.first("MVP_QA_LLM_BASE_URL", "MVP_METADATA_LLM_BASE_URL") or None
    if not api_key or not model:
        logger.info("qa.generate.llm_disabled reason=missing_api_key_or_model")
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        logger.warning("qa.generate.llm_disabled reason=%s", exc)
        return None

    completion_reserve = settings.integer(
        "MVP_QA_COMPLETION_RESERVE_TOKENS",
        1024,
        positive=True,
    )
    context_limit = settings.integer("MVP_QA_LLM_CONTEXT_TOKENS", 32768, positive=True)
    if completion_reserve >= context_limit:
        raise ValueError("QA completion reserve must be smaller than model context limit")
    return OpenAIChat(
        client=OpenAI(api_key=api_key, base_url=base_url),
        model=model,
        max_tokens=completion_reserve,
        temperature=settings.float("MVP_QA_LLM_TEMPERATURE", 0.2),
        extra_body=thinking_extra_body(base_url, settings.text("MVP_QA_LLM_THINKING"), "qa"),
        context_limit_tokens=context_limit,
    )


# Compatibility for callers that imported the old factory name.
def build_answer_llm_from_env() -> ChatModel | None:
    return build_answer_model_from_env()


class QAService:
    def __init__(
        self,
        answer_llm: AnswerLLM | ChatModel | None = None,
        query_rewriter: QueryRewriteService | None = None,
    ) -> None:
        self._retriever: HybridRouter | None = None
        self.citation_service = CitationService()
        if answer_llm is None:
            configured = build_answer_model_from_env()
            self.answer_model: ChatModel = configured or _ConfiguredChatModel(None)
            self._llm_enabled = configured is not None
        elif all(
            hasattr(answer_llm, name)
            for name in (
                "complete",
                "count_tokens",
                "context_limit_tokens",
                "completion_reserve_tokens",
                "model_name",
            )
        ):
            self.answer_model = answer_llm  # type: ignore[assignment]
            self._llm_enabled = True
        else:
            self.answer_model = _ConfiguredChatModel(answer_llm)  # type: ignore[arg-type]
            self._llm_enabled = True
        self.answer_llm = self.answer_model.complete if self._llm_enabled else None
        if (
            self.answer_model.context_limit_tokens <= 0
            or self.answer_model.completion_reserve_tokens <= 0
            or self.answer_model.completion_reserve_tokens
            >= self.answer_model.context_limit_tokens
        ):
            raise ValueError("invalid QA model context/completion token capabilities")
        self.context_top_k = settings.integer("MVP_QA_CONTEXT_TOP_K", 5, positive=True)
        self.evidence_window_tokens = settings.integer(
            "MVP_QA_EVIDENCE_WINDOW_TOKENS", 768, positive=True
        )
        self.prompt_safety_tokens = settings.integer(
            "MVP_QA_PROMPT_SAFETY_TOKENS", 256, positive=True
        )
        self.window_builder = ContextWindowBuilder()
        self.query_rewriter = query_rewriter or QueryRewriteService(
            self.answer_model if self._llm_enabled else None,
            enabled=settings.bool("MVP_QUERY_REWRITE_ENABLED", True),
            max_history_turns=settings.integer(
                "MVP_QUERY_REWRITE_MAX_HISTORY_TURNS", 6, positive=True
            ),
            max_history_tokens=settings.integer(
                "MVP_QUERY_REWRITE_MAX_HISTORY_TOKENS", 2048, positive=True
            ),
            prompt_safety_tokens=settings.integer(
                "MVP_QUERY_REWRITE_PROMPT_SAFETY_TOKENS", 128, min_value=0
            ),
            max_query_tokens=settings.integer(
                "MVP_QUERY_REWRITE_MAX_QUERY_TOKENS", 256, positive=True
            ),
        )
        self.last_budget_trace: dict = {}

    @property
    def retriever(self) -> HybridRouter:
        if self._retriever is None:
            self._retriever = HybridRouter()
        return self._retriever

    @retriever.setter
    def retriever(self, value: HybridRouter) -> None:
        self._retriever = value

    def _context_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks[: self.context_top_k]

    def _evidence_messages(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[dict[str, str]]:
        evidences: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            section = " > ".join(str(part) for part in chunk.section_path if str(part).strip())
            header = [f"[{index}]", f"doc={chunk.doc_name or chunk.doc_id or 'unknown'}"]
            if section:
                header.append(f"section={section}")
            header.append(f"score={chunk.score:.4f}")
            evidences.append(" ".join(header) + "\n" + chunk.content.strip())
        knowledge = "\n\n".join(evidences)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索证据：\n{knowledge}\n\n请基于以上证据生成完整回答。",
            },
        ]

    def _budgeted_evidence(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        prompt_limit = (
            self.answer_model.context_limit_tokens
            - self.answer_model.completion_reserve_tokens
            - self.prompt_safety_tokens
        )
        if prompt_limit <= 0:
            raise ValueError("QA model configuration leaves no prompt token budget")
        candidates = self.window_builder.candidates(
            chunks,
            top_k=self.context_top_k,
            max_window_tokens=self.evidence_window_tokens,
        )
        accepted: list[RetrievedChunk] = []
        dropped: list[str] = []
        fixed_prompt_tokens = self.answer_model.count_tokens(
            self._evidence_messages(question, [])
        )
        for candidate in candidates:
            trial = [*accepted, candidate]
            if self.answer_model.count_tokens(self._evidence_messages(question, trial)) <= prompt_limit:
                accepted = trial
                continue
            base_tokens = self.answer_model.count_tokens(
                self._evidence_messages(question, accepted)
            )
            remaining = max(1, prompt_limit - base_tokens - 24)
            try:
                smaller = self.window_builder.anchored(candidate, remaining)
            except ValueError:
                dropped.append(candidate.chunk_id)
                continue
            trial = [*accepted, smaller]
            if self.answer_model.count_tokens(self._evidence_messages(question, trial)) <= prompt_limit:
                accepted = trial
            else:
                dropped.append(candidate.chunk_id)
        if not accepted:
            raise ServiceError("QA_CONTEXT_BUDGET_EXHAUSTED", "模型上下文不足以容纳命中证据")
        used = self.answer_model.count_tokens(self._evidence_messages(question, accepted))
        self.last_budget_trace = {
            "model": self.answer_model.model_name,
            "context_limit_tokens": self.answer_model.context_limit_tokens,
            "completion_reserve_tokens": self.answer_model.completion_reserve_tokens,
            "prompt_safety_tokens": self.prompt_safety_tokens,
            "prompt_limit_tokens": prompt_limit,
            "fixed_prompt_tokens": fixed_prompt_tokens,
            "evidence_budget_tokens": max(0, prompt_limit - fixed_prompt_tokens),
            "prompt_used_tokens": used,
            "evidence_count": len(accepted),
            "dropped_chunk_ids": dropped,
        }
        return accepted

    def _generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        messages = self._evidence_messages(question, chunks)
        if self._llm_enabled:
            try:
                response = self.answer_model.complete(messages)
                if isinstance(response, str) and response.strip():
                    return response.strip()
            except Exception as exc:
                logger.warning("qa.generate.llm_failed question=%r reason=%s", question, exc)
        top = chunks[0].content if chunks else ""
        return f"基于检索证据，答案如下：{top[:120]}{' [1]' if chunks else ''}"

    def query(
        self,
        question: str,
        retrieval_config: RetrievalConfig | dict | None = None,
        *,
        history: Sequence[dict[str, str]] | None = None,
    ) -> dict:
        original_question = question.strip()
        if not original_question:
            raise ServiceError("INVALID_QUESTION", "question 不能为空")
        try:
            config = build_retrieval_config(retrieval_config)
        except ValueError as exc:
            raise ServiceError("INVALID_RETRIEVAL_CONFIG", str(exc)) from exc
        rewrite = self.query_rewriter.rewrite(original_question, history)
        effective_question = rewrite.standalone_query
        chunks = self.retriever.retrieve(effective_question, retrieval_config=config)
        if not chunks:
            raise ServiceError("NO_RETRIEVED_CHUNKS", "未检索到可用证据，请先上传并解析文档")
        evidence_chunks = self._budgeted_evidence(effective_question, chunks)
        answer = self._generate_answer(effective_question, evidence_chunks)
        payload = self.citation_service.build(answer=answer, chunks=evidence_chunks)
        payload["trace"].update(getattr(self.retriever, "last_trace", {}))
        payload["trace"]["query_rewrite"] = rewrite.trace()
        payload["trace"]["qa_budget"] = self.last_budget_trace
        payload["retrieved_chunks"] = [retrieved_chunk_payload(chunk) for chunk in chunks]
        logger.info(
            "qa.query.done question=%r config=%s citations=%d budget=%s",
            effective_question,
            asdict(config),
            len(payload.get("citations", [])),
            self.last_budget_trace,
        )
        return payload
