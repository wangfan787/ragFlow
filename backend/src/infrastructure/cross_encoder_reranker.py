from __future__ import annotations

import logging
import math
import time
from typing import Any, Protocol

from backend.src.config.settings import settings
from backend.src.contracts import Reranker
from backend.src.retrieval.ranking import chunk_order

logger = logging.getLogger("mvp_api")

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderPredictor(Protocol):
    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


def _prediction_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if expected_count == 1 and isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]
    if not isinstance(raw_scores, (list, tuple)):
        raise ValueError("cross-encoder returned an unsupported score container")

    scores: list[float] = []
    for raw_score in raw_scores:
        if isinstance(raw_score, (list, tuple)):
            if len(raw_score) != 1:
                raise ValueError("cross-encoder must return exactly one relevance score per pair")
            raw_score = raw_score[0]
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("cross-encoder returned a non-finite relevance score")
        scores.append(min(1.0, max(0.0, score)))

    if len(scores) != expected_count:
        raise ValueError(f"cross-encoder score count mismatch: {len(scores)} != {expected_count}")
    return scores


class CrossEncoderReranker(Reranker):
    """Rerank fused candidates with a lazily loaded sentence-transformers model."""

    backend_name = "cross-encoder"

    def __init__(
        self,
        predictor: CrossEncoderPredictor | None = None,
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = (
            model_name or settings.text("MVP_RERANK_MODEL") or DEFAULT_CROSS_ENCODER_MODEL
        )
        self.batch_size = (
            batch_size
            if batch_size is not None
            else settings.integer("MVP_RERANK_BATCH_SIZE", 16, positive=True)
        )
        self.max_length = (
            max_length
            if max_length is not None
            else settings.integer("MVP_RERANK_MAX_LENGTH", 512, positive=True)
        )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        self.device = device if device is not None else settings.text("MVP_RERANK_DEVICE")
        self._predictor = predictor

    def _load_predictor(self) -> CrossEncoderPredictor:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                "cross-encoder backend requires the rerank extra; run `uv sync --extra rerank`"
            ) from exc

        logger.info(
            "rerank.cross_encoder.loading model=%s max_length=%d device=%s",
            self.model_name,
            self.max_length,
            self.device or "auto",
        )
        return CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device or None,
            activation_fn=torch.nn.Sigmoid(),
        )

    def _model(self) -> CrossEncoderPredictor:
        if self._predictor is None:
            self._predictor = self._load_predictor()
        return self._predictor

    def _document_text(self, chunk: dict) -> str:
        return "\n".join(
            part
            for part in (
                str(chunk.get("doc_name", "")).strip(),
                " > ".join(str(value) for value in chunk.get("section_path", [])).strip(),
                str(chunk.get("content", "")).strip(),
            )
            if part
        )

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        if not query.strip():
            raise ValueError("query is required for cross-encoder reranking")

        pairs = [(query, self._document_text(chunk)) for chunk in chunks]
        started = time.perf_counter()
        raw_scores = self._model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = _prediction_scores(raw_scores, len(chunks))

        rows = [
            {
                **chunk,
                "rerank_score": score,
                "score": score,
            }
            for chunk, score in zip(chunks, scores)
        ]
        rows.sort(
            key=lambda row: (
                float(row["rerank_score"]),
                float(row.get("fused_score", 0.0)),
                float(row.get("keyword_score", 0.0)),
                -chunk_order(row),
                float(row.get("vector_score", 0.0)),
            ),
            reverse=True,
        )
        logger.info(
            "rerank.cross_encoder.complete model=%s candidates=%d elapsed_ms=%.1f",
            self.model_name,
            len(rows),
            (time.perf_counter() - started) * 1000,
        )
        return rows