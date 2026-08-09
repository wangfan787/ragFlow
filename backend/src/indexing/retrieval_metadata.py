from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from backend.src.config.settings import settings
from backend.src.infrastructure.openai_chat import OpenAIChat, thinking_extra_body
from backend.src.retrieval.metadata_fields import as_text_list, dedupe_text_list
from backend.src.retrieval.vectorizer import query_terms, tokenize

logger = logging.getLogger("mvp_api")

MetadataLLM = Callable[[Sequence[dict[str, str]]], str]


def metadata_llm_extra_body(base_url: str | None) -> dict:
    return thinking_extra_body(
        base_url,
        settings.text("MVP_METADATA_LLM_THINKING"),
        "metadata",
    )


def _json_from_response(text: str) -> dict:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL)
    if fenced:
        clean = fenced.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    data = json.loads(clean)
    return data if isinstance(data, dict) else {}


class RetrievalMetadataGenerator:
    """Generate RAGFlow-style retrieval metadata for chunks."""

    def __init__(self, llm_client: MetadataLLM | None = None) -> None:
        self.llm_client = llm_client

    @classmethod
    def from_env(cls) -> "RetrievalMetadataGenerator":
        api_key = settings.text("MVP_METADATA_LLM_API_KEY")
        model = settings.text("MVP_METADATA_LLM_MODEL")
        base_url = settings.text("MVP_METADATA_LLM_BASE_URL") or None
        if not api_key or not model:
            return cls()

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            logger.warning("metadata.llm_disabled reason=%s", exc)
            return cls()

        max_tokens = settings.integer("MVP_METADATA_LLM_MAX_TOKENS", 512, positive=True)
        extra_body = metadata_llm_extra_body(base_url)
        adapter = OpenAIChat(
            client=OpenAI(api_key=api_key, base_url=base_url),
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            extra_body=extra_body,
        )
        return cls(adapter.complete)

    def _prompt(self, doc_name: str, section_path: list[str], content: str) -> str:
        section = " > ".join(section_path)
        return (
            "Extract retrieval metadata for a RAG chunk.\n"
            "Return a JSON object only, without markdown fences or explanation.\n"
            'Schema: {"important_kwd":["keyword"],"question_kwd":["question"]}\n'
            "important_kwd: 3-12 concise keywords or phrases from the chunk topic.\n"
            "question_kwd: 2-6 likely user questions this chunk can answer.\n"
            f"Document: {doc_name}\n"
            f"Section: {section}\n"
            f"Content:\n{content[:2000]}"
        )

    def _messages(
        self,
        doc_name: str,
        section_path: list[str],
        content: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Extract compact retrieval metadata as JSON only. "
                    "Do not reason step by step. Do not explain."
                ),
            },
            {"role": "user", "content": self._prompt(doc_name, section_path, content)},
        ]

    def _llm_metadata(self, doc_name: str, section_path: list[str], content: str) -> dict | None:
        if not self.llm_client:
            return None
        try:
            data = _json_from_response(
                self.llm_client(self._messages(doc_name, section_path, content))
            )
        except Exception as exc:
            logger.warning(
                "metadata.llm_failed doc=%r section=%s reason=%s",
                doc_name,
                section_path,
                exc,
            )
            return None

        important_kwd = dedupe_text_list(data.get("important_kwd") or data.get("keywords"), 12)
        question_kwd = dedupe_text_list(data.get("question_kwd") or data.get("questions"), 6)
        if not important_kwd and not question_kwd:
            return None
        return {
            "important_kwd": important_kwd,
            "question_kwd": question_kwd,
            "metadata_trace": {"metadata_source": "llm"},
        }

    def _phrase_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        for raw in re.split(r"[\n\r\t,，;；、。:：|/\\()\[\]{}<>]+", text):
            clean = re.sub(r"\s+", " ", raw).strip(" -_#*")
            if not clean:
                continue
            ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+#.:-]{1,}", clean)
            candidates.extend(ascii_terms)
            for phrase in re.findall(r"[\u4e00-\u9fff]{2,12}", clean):
                candidates.append(phrase)
        return candidates

    def _fallback_keywords(self, doc_name: str, section_path: list[str], content: str) -> list[str]:
        weighted = Counter()
        structural_fields = [
            (Path(doc_name).stem, 3.0),
            (" ".join(section_path), 2.5),
        ]
        for text, field_weight in structural_fields:
            for phrase in self._phrase_candidates(text):
                weighted[phrase.lower()] += field_weight * 2.0
            for term in query_terms(text):
                if re.fullmatch(r"[\u4e00-\u9fff]+", term):
                    continue
                weighted[term] += field_weight
        if not weighted:
            for phrase in self._phrase_candidates(content):
                weighted[phrase.lower()] += 1.0
            for term in query_terms(content):
                if re.fullmatch(r"[\u4e00-\u9fff]+", term):
                    continue
                weighted[term] += 0.5
        ranked = [term for term, _ in weighted.most_common()]
        return dedupe_text_list(ranked, 12)

    def _fallback_questions(
        self,
        doc_name: str,
        section_path: list[str],
        important_kwd: list[str],
    ) -> list[str]:
        title = Path(doc_name).stem if doc_name else ""
        leaf = section_path[-1] if section_path else ""
        topic_parts = [part for part in [title, leaf] if part]
        topic = " ".join(topic_parts).strip() or " ".join(important_kwd[:3])
        if not topic:
            return []
        questions = [
            f"介绍 {topic}",
            f"{topic} 是什么",
            f"{topic} 有哪些要点",
        ]
        if important_kwd:
            questions.append(" ".join(important_kwd[:5]))
        return dedupe_text_list(questions, 6)

    def generate(self, doc_name: str, chunk: dict) -> dict:
        section_path = as_text_list(chunk.get("section_path", []))
        content = str(chunk.get("content") or chunk.get("text") or "")
        llm = self._llm_metadata(doc_name, section_path, content)
        if llm is not None:
            important_kwd = llm["important_kwd"] or self._fallback_keywords(
                doc_name,
                section_path,
                content,
            )
            question_kwd = llm["question_kwd"] or self._fallback_questions(
                doc_name,
                section_path,
                important_kwd,
            )
            trace = llm["metadata_trace"]
            trace["fallback_for_missing"] = {
                "important_kwd": not bool(llm["important_kwd"]),
                "question_kwd": not bool(llm["question_kwd"]),
            }
        else:
            important_kwd = self._fallback_keywords(doc_name, section_path, content)
            question_kwd = self._fallback_questions(doc_name, section_path, important_kwd)
            trace = {"metadata_source": "rule"}

        important_tks = tokenize(" ".join(important_kwd))
        question_tks = tokenize("\n".join(question_kwd))
        title_tks = tokenize(Path(doc_name).stem if doc_name else "")
        return {
            "important_kwd": important_kwd,
            "important_tks": important_tks,
            "question_kwd": question_kwd,
            "question_tks": question_tks,
            "title_tks": title_tks,
            "retrieval_metadata_trace": trace,
        }