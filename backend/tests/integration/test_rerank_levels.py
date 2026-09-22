"""验证重排粒度、父块聚合、失败回退及证据来源的行为。"""

from types import SimpleNamespace

from langchain_core.documents import Document
import pytest

from backend.src.apps.services.qa_service import QAService
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.config.settings import Settings, settings
from backend.src.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from backend.src.retrieval.hybrid_router import HybridRouter


class ParentStore:
    def __init__(self, parents):
        self.parents = parents
        self.lookups = []

    def query_by_ids(self, ids):
        self.lookups.append(ids)
        return [p for p in self.parents if p.metadata["chunk_id"] in ids]


class ScoredPredictor:
    def __init__(self, scores, *, fail=False):
        self.scores = scores
        self.fail = fail
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        if self.fail:
            raise RuntimeError("model unavailable")
        return [self.scores[text] for _, text in pairs]


def setup_router(*, missing_parent=False, fail=False):
    body_a, body_b, body_c = "时薪介绍。", "甲厂正在招工。", "另一个候选。"
    parents = [
        Document(page_content=body_a + body_b, metadata={
            "chunk_id": "p1", "doc_id": "d1", "chunk_role": "parent", "child_ids": ["a", "b"],
            "source_span": {"start_char": 0, "end_char": len(body_a + body_b), "accuracy": "exact"},
        }),
        Document(page_content=body_c, metadata={
            "chunk_id": "p2", "doc_id": "d2", "chunk_role": "parent", "child_ids": ["c"],
        }),
    ]
    children = []
    for cid, pid, doc, text, start, score in (
        ("a", "p1", "d1", body_a, 0, .9),
        ("b", "p1", "d1", body_b, len(body_a), .8),
        ("c", "p2", "d2", body_c, 0, .7),
    ):
        children.append({
            "chunk_id": cid, "parent_id": pid, "doc_id": doc, "content": text,
            "chunk_role": "child", "retrieval_eligible": True,
            "score": score, "vector_score": score, "keyword_score": 0., "fused_score": score,
            "parent_char_start": start, "parent_char_end": start + len(text),
            "source_span": {"start_char": start, "end_char": start + len(text), "accuracy": "exact"},
            "source_block_ids": [f"{doc}_block"],
        })
    store = ParentStore(parents[:1] if missing_parent else parents)
    router = HybridRouter(store=store)
    router.embedding_retriever.retrieve = lambda *args, **kwargs: [dict(c) for c in children]
    router.keyword_retriever.retrieve = lambda *args, **kwargs: []
    predictor = ScoredPredictor({body_a: .2, body_b: .95, body_c: .6, body_a + body_b: .9}, fail=fail)
    router._cross_encoder = CrossEncoderReranker(predictor=predictor, model_name="fixture-model")
    return router, store, predictor, children, parents


def config(**overrides):
    return RetrievalConfig(**{
        "retrieval_mode": "vector", "candidate_top_k": 3, "top_k": 2,
        "rerank_top_n": 3, "rerank_enabled": True, "rerank_backend": "cross-encoder",
        "similarity_threshold": 0., **overrides,
    })


@pytest.mark.parametrize("aggregation,expected,score", [
    ("mean", ["p2", "p1"], .575), ("max", ["p1", "p2"], .95),
])
def test_child_aggregation_changes_parent_order_without_losing_hits(aggregation, expected, score):
    router, _, predictor, _, parents = setup_router()
    chunks, trace = router.retrieve_detailed("招工", config(child_score_aggregation=aggregation))
    assert [d.metadata["chunk_id"] for d in chunks] == expected
    family = next(d for d in chunks if d.metadata["chunk_id"] == "p1")
    assert family.metadata["score"] == pytest.approx(score)
    assert family.page_content == parents[0].page_content
    assert [c["chunk_id"] for c in family.metadata["matched_children"]] == ["b", "a"]
    assert family.metadata["primary_matched_child_id"] == "b"
    assert len(predictor.calls[0]) == trace["rerank_sent_count"] == 3
    assert trace["parent_expansion"]["family_score"] == aggregation


@pytest.mark.parametrize("aggregation", ["mean", "max"])
def test_parent_reranks_unique_full_bodies_and_preserves_child_provenance(aggregation):
    router, store, predictor, children, parents = setup_router()
    chunks, trace = router.retrieve_detailed("招工", config(rerank_level="parent", child_score_aggregation=aggregation))
    assert predictor.calls == [[("招工", p.page_content) for p in parents]]
    assert len(store.lookups) == 1
    assert chunks[0].metadata["score"] == .9
    assert chunks[0].metadata["rerank_score"] == .9
    assert [c["score"] for c in chunks[0].metadata["matched_children"]] == [.9, .8]
    for hit, original in zip(chunks[0].metadata["matched_children"], children):
        assert hit["source_span"] == original["source_span"]
        assert chunks[0].page_content[hit["parent_char_start"]:hit["parent_char_end"]] == hit["snippet"]
    assert trace["rerank_level_effective"] == "parent"
    assert trace["child_score_aggregation_effective"] is None
    assert trace["parent_expansion"]["family_score"] == "cross-encoder"
    assert trace["rerank_child_pool_count"] == 3
    assert trace["rerank_sent_count"] == 2


def test_parent_pool_is_bounded_by_children_not_refilled_to_n_parents():
    router, _, predictor, _, parents = setup_router()
    chunks, trace = router.retrieve_detailed("招工", config(rerank_level="parent", rerank_top_n=2))
    assert predictor.calls == [[("招工", parents[0].page_content)]]
    assert [d.metadata["chunk_id"] for d in chunks] == ["p1"]
    assert trace["rerank_child_pool_count"] == 2
    assert trace["rerank_sent_count"] == 1


def test_parent_threshold_uses_model_score_instead_of_prefiltering_children():
    router, _, _, children, _ = setup_router()
    for child in children:
        child.update(score=.01, vector_score=.01, fused_score=.01)
    chunks, trace = router.retrieve_detailed("招工", config(rerank_level="parent", similarity_threshold=.8))
    assert [d.metadata["chunk_id"] for d in chunks] == ["p1"]
    assert trace["filtered_chunks"][0]["chunk_id"] == "p2"
    assert trace["filtered_chunks"][0]["score"] == .6


def test_missing_parent_reranks_primary_child_and_records_fallback():
    router, _, predictor, _, _ = setup_router(missing_parent=True)
    chunks, trace = router.retrieve_detailed("招工", config(rerank_level="parent"))
    assert len(predictor.calls[0]) == 2
    missing = next(d for d in chunks if d.metadata["chunk_id"] == "c")
    assert missing.metadata["chunk_role"] == "child"
    assert [c["chunk_id"] for c in missing.metadata["matched_children"]] == ["c"]
    assert trace["parent_expansion"]["parent_lookup_failed"] == 1
    assert trace["rerank_fallback_child_count"] == 1


@pytest.mark.parametrize("level", ["child", "parent"])
@pytest.mark.parametrize("aggregation,expected", [("mean", .85), ("max", .9)])
def test_failure_matches_disabled_retrieval_with_selected_aggregation(level, aggregation, expected):
    router, _, _, _, _ = setup_router(fail=True)
    options = dict(rerank_level=level, child_score_aggregation=aggregation, rerank_top_n=2)
    chunks, trace = router.retrieve_detailed("招工", config(**options))
    baseline, _ = router.retrieve_detailed("招工", config(**options, rerank_enabled=False))
    assert chunks == baseline
    assert chunks[0].metadata["score"] == pytest.approx(expected)
    assert len(chunks) == 2  # 失败回退恢复完整融合池，包括送排池外的 p2。
    assert trace["rerank_level_effective"] is None
    assert trace["child_score_aggregation_effective"] == aggregation
    assert trace["fallback_reason"].startswith("rerank_failed:")


def test_empty_parent_pool_does_not_load_reranker():
    router, _, predictor, children, _ = setup_router()
    children.clear()
    chunks, trace = router.retrieve_detailed("招工", config(rerank_level="parent"))
    assert chunks == [] and predictor.calls == []
    assert trace["rerank_sent_count"] == 0
    assert trace["rerank_level_effective"] is None


def test_parent_rank_flows_through_qa_window_and_citation():
    router, _, _, _, parents = setup_router()
    model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="甲厂正在招工 [1]"))
    service = QAService(model=model)
    service.retriever = router
    payload = service.query("哪里招工", retrieval_config=config(rerank_level="parent"))
    citation = payload["citations"][0]
    assert citation["chunk_id"] == "p1"
    assert citation["context_snippet"] == parents[0].page_content
    assert citation["primary_matched_child_id"] == "a"
    assert [c["chunk_id"] for c in citation["matched_children"]] == ["a", "b"]
    assert payload["trace"]["config"]["retrieval_config"]["rerank_level"] == "parent"


@pytest.mark.parametrize("overrides", [
    {"rerank_level": "document"}, {"child_score_aggregation": "sum"},
    {"rerank_level": True}, {"child_score_aggregation": None},
    {"rerank_enabled": True, "rerank_level": "parent", "rerank_backend": "rule"},
])
def test_invalid_rerank_combinations_are_rejected(overrides):
    with pytest.raises(ValueError):
        build_retrieval_config(overrides)


def test_rerank_options_follow_yaml_and_request_overrides(monkeypatch):
    yaml = Settings(local_path=None, overrides={"retrieval": {
        "rerank_level": "parent", "child_score_aggregation": "max", "rerank_backend": "cross-encoder",
    }})
    monkeypatch.setattr(settings, "_data", yaml._data)
    assert RetrievalConfig().rerank_level == "parent"
    assert RetrievalConfig().child_score_aggregation == "max"
    override = build_retrieval_config({"rerank_level": " CHILD ", "child_score_aggregation": "MEAN"})
    assert override.rerank_level == "child" and override.child_score_aggregation == "mean"
