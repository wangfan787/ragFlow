from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from backend.src.contracts import (
    EmbeddingModel,
    EmbeddingVector,
    validate_embedding_vector,
)

logger = logging.getLogger("mvp_api")


def embedding_cache_namespace(model: EmbeddingModel) -> str:
    """Identify vectors that are safe to reuse for the wrapped model configuration."""
    configured_dimension = getattr(model, "dimensions", None)
    if configured_dimension is None:
        configured_dimension = getattr(model, "dim", None)
    signature = {
        "adapter": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "backend": str(getattr(model, "backend_name", "")),
        "model": str(getattr(model, "model_name", "")),
        "dimension": configured_dimension or "default",
    }
    return json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class CachedEmbedding:
    """Memory + SQLite cache for any EmbeddingModel implementation."""

    def __init__(
        self,
        inner: EmbeddingModel,
        *,
        cache_path: Path,
        batch_size: int = 16,
        namespace: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._inner = inner
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.namespace = namespace or embedding_cache_namespace(inner)
        self.backend_name = str(getattr(inner, "backend_name", ""))
        self.max_input_tokens = getattr(inner, "max_input_tokens", None)
        self.dimensions = getattr(inner, "dimensions", getattr(inner, "dim", None))
        self._memory: dict[str, EmbeddingVector] = {}
        self._initialize()

    @property
    def model_name(self) -> str:
        return str(getattr(self._inner, "model_name", ""))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cache_path), timeout=30)

    def _initialize(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    namespace TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    PRIMARY KEY (namespace, text_hash)
                )
                """
            )

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_persisted(self, texts: list[str]) -> None:
        hashes = {self._text_hash(text): text for text in texts}
        hash_values = list(hashes)
        for start in range(0, len(hash_values), 500):
            batch = hash_values[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT text_hash, text, vector, dimension
                    FROM embedding_cache
                    WHERE namespace = ? AND text_hash IN ({placeholders})
                    """,
                    [self.namespace, *batch],
                ).fetchall()
            for text_hash, stored_text, raw_vector, stored_dimension in rows:
                expected_text = hashes.get(str(text_hash))
                if expected_text != stored_text:
                    continue
                try:
                    vector = [float(value) for value in json.loads(raw_vector)]
                    validate_embedding_vector(vector)
                    if len(vector) != int(stored_dimension):
                        raise ValueError("cached embedding dimension mismatch")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "embedding.cache_invalid path=%s text_hash=%s reason=%s",
                        self.cache_path,
                        text_hash,
                        exc,
                    )
                    continue
                self._memory[stored_text] = vector

    def _persist(self, rows: list[tuple[str, EmbeddingVector]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO embedding_cache (
                    namespace, text_hash, text, vector, dimension
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, text_hash) DO UPDATE SET
                    text = excluded.text,
                    vector = excluded.vector,
                    dimension = excluded.dimension
                """,
                [
                    (
                        self.namespace,
                        self._text_hash(text),
                        text,
                        json.dumps(vector, ensure_ascii=False, separators=(",", ":")),
                        len(vector),
                    )
                    for text, vector in rows
                ],
            )

    def _missing_texts(self, texts: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(text for text in texts if text not in self._memory))

    def _ensure_cached(self, texts: Sequence[str]) -> None:
        missing = self._missing_texts(texts)
        if not missing:
            return

        self._load_persisted(missing)
        missing = self._missing_texts(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._inner.encode(batch)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embedding response count mismatch: {len(vectors)} != {len(batch)}"
                )
            cache_rows: list[tuple[str, EmbeddingVector]] = []
            for text, raw_vector in zip(batch, vectors):
                vector = [float(value) for value in raw_vector]
                validate_embedding_vector(vector)
                self._memory[text] = vector
                cache_rows.append((text, vector))
            self._persist(cache_rows)

    def warm(self, texts: Sequence[str]) -> None:
        self._ensure_cached(texts)

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        self._ensure_cached(texts)
        return [list(self._memory[text]) for text in texts]
