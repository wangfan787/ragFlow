from __future__ import annotations

from dataclasses import dataclass

from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.retrieval.metadata_fields import as_text_list


@dataclass(frozen=True)
class EmbeddingTextProfile:
    name: str = "metadata-enhanced-v1"
    input_budget_tokens: int = 3072
    title_budget_tokens: int = 64
    section_budget_tokens: int = 96
    questions_budget_tokens: int = 256
    include_title: bool = True
    include_section: bool = True
    include_questions: bool = True


@dataclass(frozen=True)
class EmbeddingTextResult:
    text: str
    token_count: int
    profile: str
    removed_features: tuple[str, ...] = ()


class EmbeddingTextBuilder:
    """Build bounded embedding input while treating Child content as mandatory."""

    def __init__(
        self,
        profile: EmbeddingTextProfile | None = None,
        counter: SimpleTokenCounter | None = None,
    ) -> None:
        self.profile = profile or EmbeddingTextProfile()
        self.counter = counter or SimpleTokenCounter()

    def _bounded(self, text: str, budget: int) -> str:
        return self.counter.truncate(text, budget).strip() if text and budget > 0 else ""

    def build(self, chunk: dict, doc_name: str | None = None) -> EmbeddingTextResult:
        input_budget = int(
            chunk.get("embedding_input_budget") or self.profile.input_budget_tokens
        )
        if input_budget <= 0 or input_budget > self.profile.input_budget_tokens:
            raise ValueError("chunk embedding_input_budget is outside the approved profile")
        content = str(chunk.get("content") or chunk.get("text") or "").strip()
        if not content:
            raise ValueError(f"embedding content is empty for chunk {chunk.get('chunk_id')}")
        content_tokens = self.counter.count(content)
        if content_tokens > input_budget:
            raise ValueError(
                f"child content exceeds embedding budget chunk_id={chunk.get('chunk_id')} "
                f"tokens={content_tokens} limit={input_budget}"
            )

        title = self._bounded(
            str(chunk.get("doc_name") or doc_name or ""), self.profile.title_budget_tokens
        )
        section = self._bounded(
            " > ".join(as_text_list(chunk.get("section_path", []))),
            self.profile.section_budget_tokens,
        )
        questions = self._bounded(
            "\n".join(as_text_list(chunk.get("question_kwd", []))),
            self.profile.questions_budget_tokens,
        )
        features = {
            "title": title if self.profile.include_title else "",
            "section": section if self.profile.include_section else "",
            "questions": questions if self.profile.include_questions else "",
        }

        def render() -> str:
            parts = []
            if features["title"]:
                parts.append(f"Document: {features['title']}")
            if features["section"]:
                parts.append(f"Section: {features['section']}")
            parts.append(f"Content:\n{content}")
            if features["questions"]:
                parts.append(f"Related questions:\n{features['questions']}")
            return "\n".join(parts)

        removed: list[str] = []
        text = render()
        # Approved low-to-high priority: questions, title, section. Content is
        # never truncated here; a content-only overflow is an upstream error.
        for feature in ("questions", "title", "section"):
            if self.counter.count(text) <= input_budget:
                break
            if features[feature]:
                features[feature] = ""
                removed.append(feature)
                text = render()
        token_count = self.counter.count(text)
        if token_count > input_budget:
            raise ValueError(
                f"embedding input exceeds budget after optional feature removal "
                f"chunk_id={chunk.get('chunk_id')} tokens={token_count}"
            )
        return EmbeddingTextResult(text, token_count, self.profile.name, tuple(removed))
