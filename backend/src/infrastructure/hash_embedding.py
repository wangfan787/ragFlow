from __future__ import annotations

import hashlib
from typing import Sequence

from backend.src.contracts import EmbeddingModel, EmbeddingVector, validate_embedding_vector
from backend.src.retrieval.vectorizer import tokenize


class HashEmbedding(EmbeddingModel):
    backend_name = "hash"
    model_name = "hash-v1"
    max_input_tokens = None

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.dimensions = dim

    def _vector_for_text(self, text: str) -> EmbeddingVector:
        values = [0.0] * self.dim
        terms = tokenize(text)
        if not terms:
            terms = [text.strip().lower() or "__empty__"]
        for term in terms:
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = digest[0] % self.dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            values[index] += sign
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        validate_embedding_vector(values)
        return values

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self._vector_for_text(text) for text in texts]
