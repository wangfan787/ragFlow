"""单轮问答：检索 → 命中窗口 → 完整消息预算 → LangChain → 引用。"""

from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel

from backend.src.apps.services.common_service import ServiceError
from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.chunking.token_counter import count_message_tokens
from backend.src.citation.citation_service import CitationService
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.config.settings import settings
from backend.src.infrastructure.models import build_chat
from backend.src.retrieval.hybrid_router import HybridRouter
from langchain_core.documents import Document

_SYSTEM_PROMPT = (
    "只依据给定证据回答；在事实后用 [编号] 标注来源；"
    "证据不足时仅回答‘未找到相关依据’；不要虚构来源。"
    "图片描述是模型提取结果，无法确认的细节不要猜测。"
)


class QAService:
    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.model = model
        self._retriever: HybridRouter | None = None
        self.citation_service = CitationService()
        self.window_builder = ContextWindowBuilder()
        self.context_limit_tokens = settings.integer("MVP_QA_LLM_CONTEXT_TOKENS", 32768, positive=True)
        self.completion_reserve_tokens = settings.integer("MVP_QA_COMPLETION_RESERVE_TOKENS", 1024, positive=True)
        self.prompt_safety_tokens = settings.integer("MVP_QA_PROMPT_SAFETY_TOKENS", 256, min_value=0)
        self.context_top_k = settings.integer("MVP_QA_CONTEXT_TOP_K", 5, positive=True)
        self.evidence_window_tokens = settings.integer("MVP_QA_EVIDENCE_WINDOW_TOKENS", 384, positive=True)
        self.last_budget_trace: dict = {}

    @property
    def retriever(self) -> HybridRouter:
        if self._retriever is None:
            self._retriever = HybridRouter()
        return self._retriever

    @retriever.setter
    def retriever(self, value: HybridRouter) -> None:
        self._retriever = value

    def _evidence_messages(self, question: str, chunks: list[Document]) -> list[dict]:
        evidence = []
        for index, chunk in enumerate(chunks, start=1):
            section = " > ".join(chunk.metadata.get("section_path", []))
            evidence.append(f"[{index}] {chunk.metadata.get('doc_name') or chunk.metadata['doc_id']} / {section}\n{chunk.page_content}")
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n证据：\n" + "\n\n".join(evidence)},
        ]

    def _budgeted_evidence(self, question: str, chunks: list[Document]) -> list[Document]:
        prompt_limit = self.context_limit_tokens - self.completion_reserve_tokens - self.prompt_safety_tokens
        if prompt_limit <= 0:
            raise ValueError("QA 配置没有留下可用的证据预算")
        candidates = self.window_builder.candidates(
            chunks, top_k=self.context_top_k, max_window_tokens=self.evidence_window_tokens,
        )
        accepted: list[Document] = []
        dropped: list[str] = []
        fixed_tokens = count_message_tokens(self._evidence_messages(question, []))
        for candidate in candidates:
            trial = [*accepted, candidate]
            if count_message_tokens(self._evidence_messages(question, trial)) <= prompt_limit:
                accepted = trial
                continue
            base_tokens = count_message_tokens(self._evidence_messages(question, accepted))
            try:
                smaller = self.window_builder.anchored(candidate, max(1, prompt_limit - base_tokens - 24))
            except ValueError:
                dropped.append(candidate.metadata["chunk_id"])
                continue
            trial = [*accepted, smaller]
            if count_message_tokens(self._evidence_messages(question, trial)) <= prompt_limit:
                accepted = trial
            else:
                dropped.append(candidate.metadata["chunk_id"])
        if not accepted:
            raise ServiceError("QA_CONTEXT_BUDGET_EXHAUSTED", "模型上下文不足以容纳命中证据")
        self.last_budget_trace = {
            "context_limit_tokens": self.context_limit_tokens,
            "completion_reserve_tokens": self.completion_reserve_tokens,
            "prompt_safety_tokens": self.prompt_safety_tokens,
            "prompt_limit_tokens": prompt_limit,
            "fixed_prompt_tokens": fixed_tokens,
            "evidence_budget_tokens": max(0, prompt_limit - fixed_tokens),
            "prompt_used_tokens": count_message_tokens(self._evidence_messages(question, accepted)),
            "evidence_count": len(accepted),
            "dropped_chunk_ids": dropped,
        }
        return accepted

    def _generate_answer(self, question: str, chunks: list[Document]) -> str:
        if self.model is None:
            self.model = build_chat("qa")
        try:
            content = self.model.invoke(self._evidence_messages(question, chunks)).content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("聊天模型返回空内容")
            return content.strip()
        except Exception as exc:
            raise HTTPException(502, "聊天模型调用失败，请检查模型服务后重试") from exc

    def query(self, question: str, retrieval_config: RetrievalConfig | dict | None = None) -> dict:
        question = question.strip()
        if not question or len(question.encode("utf-8")) > 3072:
            raise ServiceError("INVALID_QUESTION", "问题不能为空且不能超过 3072 UTF-8 字节")
        config = build_retrieval_config(retrieval_config)
        chunks = self.retriever.retrieve(question, retrieval_config=config)
        if not chunks:
            raise ServiceError("NO_RETRIEVED_CHUNKS", "未检索到可用证据，请先上传并解析文档")
        evidence = self._budgeted_evidence(question, chunks)
        answer = self._generate_answer(question, evidence)
        payload = self.citation_service.build(answer=answer, chunks=evidence)
        payload["trace"].update(getattr(self.retriever, "last_trace", {}))
        payload["trace"]["qa_budget"] = self.last_budget_trace
        payload["retrieved_chunks"] = [{"content": chunk.page_content, **chunk.metadata} for chunk in chunks]
        return payload
