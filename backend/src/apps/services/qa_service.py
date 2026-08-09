from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict

from backend.src.apps.services.common_service import ServiceError
from backend.src.citation.citation_service import CitationService
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.config.settings import settings
from backend.src.infrastructure.openai_chat import (
    OpenAIChat,
    thinking_extra_body,
)
from backend.src.retrieval.hybrid_router import HybridRouter
from backend.src.retrieval.metadata_fields import retrieved_chunk_payload

logger = logging.getLogger("mvp_api")

AnswerLLM = Callable[[Sequence[dict[str, str]]], str]


def build_answer_llm_from_env() -> AnswerLLM | None:
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

    max_tokens = settings.integer("MVP_QA_LLM_MAX_TOKENS", 4096, positive=True)
    temperature = settings.float("MVP_QA_LLM_TEMPERATURE", 0.2)
    extra_body = thinking_extra_body(base_url, settings.text("MVP_QA_LLM_THINKING"), "qa")
    logger.info(
        "qa.generate.llm_enabled model=%s base_url=%s max_tokens=%d temperature=%s",
        model,
        base_url,
        max_tokens,
        temperature,
    )
    adapter = OpenAIChat(
        client=OpenAI(api_key=api_key, base_url=base_url),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    return adapter.complete


class QAService:
    def __init__(self, answer_llm: AnswerLLM | None = None) -> None:
        self._retriever: HybridRouter | None = None
        self.citation_service = CitationService()
        self.answer_llm = answer_llm if answer_llm is not None else build_answer_llm_from_env()
        self.context_top_k = settings.integer("MVP_QA_CONTEXT_TOP_K", 5, positive=True)
        self.context_chars_per_chunk = settings.integer(
            "MVP_QA_CONTEXT_CHARS_PER_CHUNK",
            2500,
            positive=True,
        )

    @property
    def retriever(self) -> HybridRouter:
        if self._retriever is None:
            self._retriever = HybridRouter()
        return self._retriever

    @retriever.setter
    def retriever(self, value: HybridRouter) -> None:
        self._retriever = value

    def _chunk_value(self, chunk, key: str, default=""):
        if isinstance(chunk, dict):
            return chunk.get(key, default)
        return getattr(chunk, key, default)

    def _context_chunks(self, chunks: list) -> list:
        return chunks[: self.context_top_k]

    def _evidence_messages(self, question: str, chunks: list) -> list[dict[str, str]]:
        evidences: list[str] = []
        for index, chunk in enumerate(self._context_chunks(chunks), start=1):
            content = str(self._chunk_value(chunk, "content", ""))
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            if len(content) > self.context_chars_per_chunk:
                content = content[: self.context_chars_per_chunk].rstrip() + "\n..."
            section_path = self._chunk_value(chunk, "section_path", []) or []
            if not isinstance(section_path, list):
                section_path = [str(section_path)]
            section = " > ".join(str(part) for part in section_path if str(part).strip())
            doc_name = str(self._chunk_value(chunk, "doc_name", "") or "")
            doc_id = str(self._chunk_value(chunk, "doc_id", "") or "")
            score = self._chunk_value(chunk, "score", None)
            header_parts = [
                f"[{index}]",
                f"doc={doc_name or doc_id or 'unknown'}",
            ]
            if section:
                header_parts.append(f"section={section}")
            if score is not None:
                header_parts.append(f"score={float(score):.4f}")
            evidences.append(" ".join(header_parts) + "\n" + content)

        knowledge = "\n\n".join(evidences)
        return [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的知识库问答助手。请只依据用户提供的检索证据回答。"
                    "需要综合多个证据，不要只摘抄第一条证据。"
                    "如果证据不足或与问题无关，请回答“未检索到相关证据”。"
                    "回答应完整、结构清晰、使用中文；必要时保留代码或表格要点。"
                    "引用证据时使用 [1]、[2] 这样的编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n检索证据：\n{knowledge}\n\n请基于以上证据生成完整回答。"
                ),
            },
        ]

    def _generate_answer(self, question: str, chunks: list) -> str:
        messages = self._evidence_messages(question, chunks)
        if self.answer_llm:
            try:
                response = self.answer_llm(messages)
                if isinstance(response, str) and response.strip():
                    logger.info(
                        "qa.generate.llm_success question=%r chunks=%d",
                        question,
                        len(chunks),
                    )
                    return response.strip()
            except Exception as exc:
                logger.warning(
                    "qa.generate.llm_failed question=%r reason=%s",
                    question,
                    exc,
                    exc_info=True,
                )

        top = str(self._chunk_value(chunks[0], "content", "")) if chunks else ""
        logger.info(
            "qa.generate.fallback question=%r top_doc=%s top_chunk=%s",
            question,
            self._chunk_value(chunks[0], "doc_id", None) if chunks else None,
            self._chunk_value(chunks[0], "chunk_id", None) if chunks else None,
        )
        citation = " [1]" if chunks else ""
        return f"基于检索证据，答案如下：{top[:120]}{citation}"

    def query(
        self,
        question: str,
        retrieval_config: RetrievalConfig | dict | None = None,
    ) -> dict:
        if not question.strip():
            raise ServiceError("INVALID_QUESTION", "question 不能为空")

        try:
            config = build_retrieval_config(retrieval_config)
        except ValueError as exc:
            raise ServiceError("INVALID_RETRIEVAL_CONFIG", str(exc)) from exc

        config_payload = asdict(config)
        logger.info("qa.query.start question=%r config=%s", question, config_payload)
        chunks = self.retriever.retrieve(question, retrieval_config=config)
        if not chunks:
            logger.warning("qa.query.no_chunks question=%r config=%s", question, config_payload)
            raise ServiceError("NO_RETRIEVED_CHUNKS", "未检索到可用证据，请先上传并解析文档")

        evidence_chunks = self._context_chunks(chunks)
        answer = self._generate_answer(question, evidence_chunks)
        payload = self.citation_service.build(answer=answer, chunks=evidence_chunks)
        payload["trace"].update(getattr(self.retriever, "last_trace", {}))
        payload["retrieved_chunks"] = [retrieved_chunk_payload(chunk) for chunk in chunks]
        logger.info(
            "qa.query.done question=%r top=%s citations=%d",
            question,
            [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 4),
                    "section_path": chunk.section_path,
                }
                for chunk in chunks[:5]
            ],
            len(payload.get("citations", [])),
        )
        return payload
