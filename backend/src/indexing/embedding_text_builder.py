"""仅嵌入完整子块正文；越界时要求重新切片，不能截断来源。"""

from dataclasses import dataclass
from langchain_core.documents import Document

from backend.src.chunking.token_counter import count_tokens


@dataclass(frozen=True)
class EmbeddingTextResult:
    text: str
    token_count: int
    profile: str = "child-body-v1"


class EmbeddingTextBuilder:
    profile = "child-body-v1"

    def build(self, chunk: Document) -> EmbeddingTextResult:
        text = chunk.page_content
        if not text.strip():
            raise ValueError(f"子块正文为空：{chunk.metadata.get('chunk_id')}")
        tokens = count_tokens(text)
        budget = min(int(chunk.metadata.get("embedding_input_budget", 192)), 192)
        if budget <= 0 or tokens > budget or len(text.encode("utf-8")) > 3072:
            raise ValueError(f"子块超过嵌入输入限制，请重新切片：{chunk.metadata.get('chunk_id')}")
        return EmbeddingTextResult(text, tokens, self.profile)
