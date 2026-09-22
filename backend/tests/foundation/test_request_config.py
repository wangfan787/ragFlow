"""请求配置的类型约束、组合约束与 YAML 默认值。"""
import pytest

from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.config.qa_config import QAConfig, build_qa_config
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.config.settings import settings


@pytest.mark.parametrize("overrides", [
    {"filters": "not-a-filter"},
    {"rerank_enabled": "garbage"},
    {"rerank_enabled": "false"},
    {"top_k": 2.5},
    {"top_k": True},
    {"vector_weight": float("nan")},
    {"rerank_top_n": ""},
])
def test_invalid_retrieval_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        build_retrieval_config(overrides)


@pytest.mark.parametrize("cls,builder", [
    (RetrievalConfig, build_retrieval_config),
    (QAConfig, build_qa_config),
    (ChunkConfig, build_chunk_config),
])
def test_validated_config_is_passed_through(cls, builder):
    config = cls()
    assert builder(config) is config


def test_request_defaults_follow_yaml_and_explicit_overrides_win(monkeypatch):
    monkeypatch.setitem(settings._data["qa"], "evidence_window_tokens", 220)
    monkeypatch.setitem(settings._data["chunking"], "child_target_tokens", 64)
    assert QAConfig().evidence_window_tokens == 220
    assert QAConfig(evidence_window_tokens=100).evidence_window_tokens == 100
    assert ChunkConfig().child_target_tokens == 64
    monkeypatch.setitem(settings._data["retrieval"], "rerank_level", "parent")
    monkeypatch.setitem(settings._data["retrieval"], "child_score_aggregation", "max")
    monkeypatch.setitem(settings._data["retrieval"], "rerank_backend", "cross-encoder")
    assert RetrievalConfig().rerank_level == "parent"
    assert RetrievalConfig().child_score_aggregation == "max"
    override = build_retrieval_config({"rerank_level": " CHILD ", "child_score_aggregation": "MEAN"})
    assert override.rerank_level == "child" and override.child_score_aggregation == "mean"


def test_retrieval_mode_validation_and_normalization():
    config = build_retrieval_config(retrieval_mode=" Vector ")
    assert config.retrieval_mode == "vector"
    with pytest.raises(ValueError, match="retrieval_mode"):
        build_retrieval_config(retrieval_mode="semantic")
    with pytest.raises(ValueError, match="unknown_field"):
        build_retrieval_config(unknown_field=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rerank_top_n": 2, "top_k": 5, "candidate_top_k": 30},   # 小于 top_k
        {"rerank_top_n": 50, "top_k": 5, "candidate_top_k": 30},  # 大于召回池
    ],
)
def test_rerank_top_n_outside_funnel_is_rejected(overrides):
    with pytest.raises(ValueError, match="rerank_top_n"):
        build_retrieval_config(overrides)


def test_qa_config_build_and_validation():
    config = build_qa_config({"evidence_mode": "Child_Only "}, evidence_window_tokens=192)
    assert config == QAConfig(context_top_k=5, evidence_mode="child_only", evidence_window_tokens=192)
    with pytest.raises(ValueError, match="evidence_mode"):
        build_qa_config(evidence_mode="everything")
    with pytest.raises(ValueError, match="window_tokens"):
        build_qa_config(window_tokens=192)
    with pytest.raises(ValueError, match="evidence_window_tokens"):
        build_qa_config(evidence_window_tokens=None)
    with pytest.raises(ValueError):
        QAConfig(context_top_k=0)
    with pytest.raises(ValueError):
        QAConfig(evidence_window_tokens=0)


@pytest.mark.parametrize("overrides", [
    {"rerank_level": "document"}, {"child_score_aggregation": "sum"},
    {"rerank_level": True}, {"child_score_aggregation": None},
    {"rerank_enabled": True, "rerank_level": "parent", "rerank_backend": "rule"},
])
def test_invalid_rerank_combinations_are_rejected(overrides):
    with pytest.raises(ValueError):
        build_retrieval_config(overrides)
