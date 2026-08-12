from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.contracts import (
    EmbeddingModel,
    EmbeddingVector,
    validate_embedding_vector,
)

logger = logging.getLogger("mvp_api")


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI-compatible text embedding adapter."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        backend_name: str = "openai-compatible",
        dimensions: int | None = None,
        batch_size: int = 64,
        max_input_tokens: int | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.backend_name = backend_name
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        if max_batch_size is not None:
            self.batch_size = min(self.batch_size, max(1, max_batch_size))
        self.max_input_tokens = max_input_tokens if max_input_tokens and max_input_tokens > 0 else None
        self._token_counter = SimpleTokenCounter()
        self.truncated_input_count = 0

    @property
    def model_name(self) -> str:
        return self.model

    def _encode_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        safe_texts: list[str] = []
        truncated = 0
        largest_input = 0
        for text in texts:
            if self.max_input_tokens is None:
                safe_texts.append(text)
                continue
            token_count = self._token_counter.count(text)
            if token_count == 0 and text.strip():
                raise RuntimeError("failed to count tokens for non-empty embedding input")
            largest_input = max(largest_input, token_count)
            if token_count > self.max_input_tokens:
                text = self._token_counter.truncate(text, self.max_input_tokens)
                truncated += 1
            safe_texts.append(text)

        if truncated:
            self.truncated_input_count += truncated
            logger.warning(
                "embedding.input_truncated backend=%s model=%s count=%d max_tokens_before=%d limit=%d",
                self.backend_name,
                self.model,
                truncated,
                largest_input,
                self.max_input_tokens,
            )

        request: dict[str, Any] = {
            "model": self.model,
            "input": safe_texts,
        }
        if self.dimensions:
            request["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**request)
        data = sorted(response.data, key=lambda item: item.index)
        indices = [item.index for item in data]
        if indices != list(range(len(safe_texts))):
            raise ValueError(f"embedding response indices mismatch: {indices}")
        vectors: list[EmbeddingVector] = []
        for item in data:
            vector = [float(value) for value in item.embedding]
            validate_embedding_vector(vector)
            vectors.append(vector)
        return vectors

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        vectors: list[EmbeddingVector] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode_batch(texts[start : start + self.batch_size]))
        if len(vectors) != len(texts):
            raise ValueError(f"embedding response count mismatch: {len(vectors)} != {len(texts)}")
        return vectors
