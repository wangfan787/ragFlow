"""单轮问答：检索 → 命中窗口 → 完整消息预算 → LangChain → 引用。

请求级可观测性（docs/plan/README.md §4.2 前置改造 4）：
- trace.config       本次请求实际生效的检索/QA 配置快照、模型名和索引名
- trace.timings_ms   retrieval / window / generation / citation / total 分阶段耗时
- trace.usage        模型真实 usage；供应商未返回的字段记 "unavailable"，不补零
所有 trace 都是请求局部对象，不再通过共享的实例状态传递。
"""

import time
from dataclasses import asdict, dataclass

from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel

from backend.src.apps.services.common_service import ServiceError
from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.chunking.token_counter import count_message_tokens
from backend.src.citation.citation_service import CitationService
from backend.src.config.qa_config import QAConfig, build_qa_config
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

# 供应商未返回字段统一使用该哨兵值，评测和报告不得把它当成 0 参与计算
UNAVAILABLE = "unavailable"


def _elapsed_ms(started: float) -> float:
    """用单调时钟计算阶段耗时（毫秒），避免系统时间跳变影响测量"""
    return (time.perf_counter() - started) * 1000.0


@dataclass(frozen=True)
class GeneratedAnswer:
    """模型生成结果 + 请求级 usage；不再只向上返回回答文本"""

    answer: str
    usage: dict


class QAService:
    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.model = model
        self._retriever: HybridRouter | None = None
        self.citation_service = CitationService()
        self.window_builder = ContextWindowBuilder()
        self.context_limit_tokens = settings.integer("MVP_QA_LLM_CONTEXT_TOKENS", 32768, positive=True)
        self.completion_reserve_tokens = settings.integer("MVP_QA_COMPLETION_RESERVE_TOKENS", 1024, positive=True)
        self.prompt_safety_tokens = settings.integer("MVP_QA_PROMPT_SAFETY_TOKENS", 256, min_value=0)
        # 全局默认值只作为"未传请求级 QAConfig"时的兜底；
        # 逐题扫描参数必须通过 query(qa_config=...) 传入。
        self.context_top_k = settings.integer("MVP_QA_CONTEXT_TOP_K", 5, positive=True)
        self.evidence_window_tokens = settings.integer("MVP_QA_EVIDENCE_WINDOW_TOKENS", 384, positive=True)

    @property
    def retriever(self) -> HybridRouter:
        if self._retriever is None:
            self._retriever = HybridRouter()
        return self._retriever

    @retriever.setter
    def retriever(self, value: HybridRouter) -> None:
        self._retriever = value

    def _default_qa_config(self) -> QAConfig:
        """未显式传 QAConfig 时，用服务实例的全局默认值构造请求级配置。"""
        return QAConfig(
            context_top_k=self.context_top_k,
            evidence_mode="window",
            evidence_window_tokens=self.evidence_window_tokens,
        )

    def _model_name(self) -> str | None:
        """尽力取聊天模型名；测试替身或未初始化时返回 None。"""
        for attr in ("model_name", "model"):
            value = getattr(self.model, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _evidence_messages(self, question: str, chunks: list[Document]) -> list[dict]:
        evidence = []
        for index, chunk in enumerate(chunks, start=1):
            section = " > ".join(chunk.metadata.get("section_path", []))
            evidence.append(f"[{index}] {chunk.metadata.get('doc_name') or chunk.metadata['doc_id']} / {section}\n{chunk.page_content}")
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n证据：\n" + "\n\n".join(evidence)},
        ]

    def _child_only_document(self, chunk: Document) -> Document:
        """child_only 模式：证据只保留主命中子块自身文本与来源信息。

        不猜测子块在父块中的范围，直接使用召回时记录的子块明细；
        没有命中明细的旧数据退化为父块文本，保证链路可用。
        """
        children = chunk.metadata.get("matched_children") or []
        primary_id = chunk.metadata.get("primary_matched_child_id") or chunk.metadata.get("matched_child_id")
        primary = next(
            (item for item in children if str(item.get("chunk_id")) == str(primary_id)),
            children[0] if children else None,
        )
        if primary is None:
            return chunk
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "chunk_id": str(primary.get("chunk_id") or chunk.metadata["chunk_id"]),
                "chunk_role": "child",
                "matched_children": [primary],
                "source_span": dict(primary.get("source_span") or {}),
                "context_span": {},
                "prompt_span": {"fallback": "child_only"},
            }
        )
        return Document(page_content=str(primary.get("snippet", "")), metadata=metadata)

    def _evidence_candidates(self, chunks: list[Document], qa_config: QAConfig) -> list[Document]:
        """按 evidence_mode 构造初始证据候选；总体预算约束仍在 _budgeted_evidence 中执行。"""
        selected = list(chunks[: qa_config.context_top_k])
        if qa_config.evidence_mode == "child_only":
            docs = [self._child_only_document(chunk) for chunk in selected]
            return [doc for doc in docs if doc.page_content.strip()]
        if qa_config.evidence_mode == "full_parent":
            # 整个父块直接作为证据；超出总预算时由预算循环降级为锚定窗口
            return selected
        return self.window_builder.candidates(
            selected,
            top_k=qa_config.context_top_k,
            max_window_tokens=qa_config.evidence_window_tokens,
        )

    def _budgeted_evidence(
        self, question: str, chunks: list[Document], qa_config: QAConfig
    ) -> tuple[list[Document], dict]:
        prompt_limit = self.context_limit_tokens - self.completion_reserve_tokens - self.prompt_safety_tokens
        if prompt_limit <= 0:
            raise ValueError("QA 配置没有留下可用的证据预算")
        candidates = self._evidence_candidates(chunks, qa_config)
        accepted: list[Document] = []
        dropped: list[str] = []
        fixed_tokens = count_message_tokens(self._evidence_messages(question, []))
        for candidate in candidates:
            trial = [*accepted, candidate]
            if count_message_tokens(self._evidence_messages(question, trial)) <= prompt_limit:
                accepted = trial
                continue
            base_tokens = count_message_tokens(self._evidence_messages(question, accepted))
            # 只有 window / full_parent 模式的候选具备父块锚点，可以缩窗重试；
            # child_only 候选没有可靠坐标，超预算直接丢弃，不猜测截断位置。
            if qa_config.evidence_mode == "child_only":
                dropped.append(candidate.metadata["chunk_id"])
                continue
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
        budget_trace = {
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
        return accepted, budget_trace

    def _extract_usage(self, message: object) -> dict:
        """从模型响应对象提取供应商真实 usage。

        优先读 LangChain 标准的 usage_metadata，其次 OpenAI 兼容的
        response_metadata.token_usage；两者都没有时各字段记 "unavailable"，
        不得用 0 冒充真实零消耗。
        """
        usage = {
            "model": self._model_name() or UNAVAILABLE,
            "input_tokens": UNAVAILABLE,
            "output_tokens": UNAVAILABLE,
            "total_tokens": UNAVAILABLE,
            # 项目尚未维护可靠价格表；接入价格配置后才能计算真实成本
            "cost": UNAVAILABLE,
        }
        metadata = getattr(message, "usage_metadata", None)
        if isinstance(metadata, dict) and metadata.get("total_tokens") is not None:
            usage.update(
                {
                    "input_tokens": metadata.get("input_tokens", UNAVAILABLE),
                    "output_tokens": metadata.get("output_tokens", UNAVAILABLE),
                    "total_tokens": metadata.get("total_tokens"),
                }
            )
            return usage
        response_metadata = getattr(message, "response_metadata", None)
        token_usage = response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None
        if isinstance(token_usage, dict) and token_usage.get("total_tokens") is not None:
            usage.update(
                {
                    "input_tokens": token_usage.get("prompt_tokens", UNAVAILABLE),
                    "output_tokens": token_usage.get("completion_tokens", UNAVAILABLE),
                    "total_tokens": token_usage.get("total_tokens"),
                }
            )
        return usage

    def _generate_answer(self, question: str, chunks: list[Document]) -> GeneratedAnswer:
        if self.model is None:
            self.model = build_chat("qa")
        try:
            message = self.model.invoke(self._evidence_messages(question, chunks))
            content = message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("聊天模型返回空内容")
            return GeneratedAnswer(answer=content.strip(), usage=self._extract_usage(message))
        except Exception as exc:
            raise HTTPException(502, "聊天模型调用失败，请检查模型服务后重试") from exc

    def query(
        self,
        question: str,
        retrieval_config: RetrievalConfig | dict | None = None,
        qa_config: QAConfig | dict | None = None,
    ) -> dict:
        question = question.strip()
        if not question or len(question.encode("utf-8")) > 3072:
            raise ServiceError("INVALID_QUESTION", "问题不能为空且不能超过 3072 UTF-8 字节")
        config = build_retrieval_config(retrieval_config)
        qa = build_qa_config(qa_config) if qa_config is not None else self._default_qa_config()

        timings: dict[str, float] = {}
        total_started = time.perf_counter()

        # ---- 阶段 1：检索（优先使用请求级 trace，避免共享 last_trace 串数据）----
        stage_started = time.perf_counter()
        detailed = getattr(self.retriever, "retrieve_detailed", None)
        if callable(detailed):
            chunks, retrieval_trace = detailed(question, retrieval_config=config)
        else:
            # 测试替身或旧实现只有 retrieve()：退回共享 last_trace（仅调试场景）
            chunks = self.retriever.retrieve(question, retrieval_config=config)
            retrieval_trace = dict(getattr(self.retriever, "last_trace", None) or {})
        timings["retrieval"] = _elapsed_ms(stage_started)
        if not chunks:
            raise ServiceError("NO_RETRIEVED_CHUNKS", "未检索到可用证据，请先上传并解析文档")

        # ---- 阶段 2：证据窗口 / 模式选择 + 总预算 ----
        stage_started = time.perf_counter()
        evidence, budget_trace = self._budgeted_evidence(question, chunks, qa)
        timings["window"] = _elapsed_ms(stage_started)

        # ---- 阶段 3：回答生成 ----
        stage_started = time.perf_counter()
        generated = self._generate_answer(question, evidence)
        timings["generation"] = _elapsed_ms(stage_started)

        # ---- 阶段 4：引用抽取 ----
        stage_started = time.perf_counter()
        payload = self.citation_service.build(answer=generated.answer, chunks=evidence)
        timings["citation"] = _elapsed_ms(stage_started)

        timings["total"] = _elapsed_ms(total_started)

        payload["trace"].update(retrieval_trace)
        payload["trace"]["qa_budget"] = budget_trace
        # 配置快照：run 文件据此证明"这一题实际用了哪组参数"，不靠环境变量反推
        payload["trace"]["config"] = {
            "retrieval_config": asdict(config),
            "qa_config": qa.as_trace_dict(),
            "model": self._model_name() or UNAVAILABLE,
            "index_name": settings.text("MVP_ELASTICSEARCH_INDEX", "rag-md-v1"),
        }
        payload["trace"]["timings_ms"] = {key: round(value, 2) for key, value in timings.items()}
        payload["trace"]["usage"] = generated.usage
        payload["retrieved_chunks"] = [{"content": chunk.page_content, **chunk.metadata} for chunk in chunks]
        return payload
