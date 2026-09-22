from __future__ import annotations

import pytest

from backend.src.infrastructure.cross_encoder_reranker import CrossEncoderReranker


def _chunk(chunk_id: str, content: str, fused_score: float = 0.5) -> dict:
    return {
        "doc_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "content": content,
        "fused_score": fused_score,
        "score": fused_score,
        "keyword_score": 0.0,
        "vector_score": fused_score,
    }


class FakePredictor:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return [0.1, 0.9]


def test_cross_encoder_reranker_scores_raw_query_document_pairs() -> None:
    predictor = FakePredictor()
    reranker = CrossEncoderReranker(predictor=predictor, model_name="fake-zh-model")
    rows = reranker.rerank(
        "退款期限",
        [
            {**_chunk("a", "无关正文"), "doc_name": "甲", "section_path": ["规则"]},
            {**_chunk("b", "退款期限是七天"), "doc_name": "乙", "section_path": ["售后"]},
        ],
    )

    assert [row["chunk_id"] for row in rows] == ["b", "a"]
    pairs, kwargs = predictor.calls[0]
    assert pairs == [
        ("退款期限", "甲\n规则\n无关正文"),
        ("退款期限", "乙\n售后\n退款期限是七天"),
    ]
    assert kwargs["batch_size"] == 16
    assert rows[0]["rerank_score"] == pytest.approx(0.9)
