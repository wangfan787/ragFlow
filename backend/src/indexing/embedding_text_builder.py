"""仅嵌入完整子块正文；越界时要求重新切片，不能截断来源。"""

from dataclasses import dataclass
from langchain_core.documents import Document

from backend.src.chunking.token_counter import count_tokens
from backend.src.config.settings import settings



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
        model_limit = settings.integer("embedding.max_input_tokens", positive=True)
        if tokens > model_limit:
            raise ValueError(
                f"子块超过模型嵌入输入上限（估算 {tokens} > {model_limit} token），"
                f"未截断原文，请按结构拆分或更换模型：{chunk.metadata.get('chunk_id')}"
            )
        if not chunk.metadata.get("preserve_structure"):
            budget = chunk.metadata["embedding_input_budget"]
            if budget <= 0 or tokens > budget:
                raise ValueError(f"子块超过正文嵌入预算，请重新切片：{chunk.metadata.get('chunk_id')}")
        return EmbeddingTextResult(text, tokens, self.profile)
