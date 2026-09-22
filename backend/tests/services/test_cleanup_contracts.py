"""简化后的边界回归：拒绝非法配置、正式模型接线、内部错误不被预算兜底隐藏。"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.src.apps.services.qa_service import QAService
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


class RecordingRouter:
    def __init__(self):
        self.questions = []

    def retrieve_detailed(self, question, retrieval_config):
        self.questions.append(question)
        return [Document(page_content="独立问题的证据。", metadata={
            "doc_id": "d", "chunk_id": "p", "score": 0.9,
        })], {}


@pytest.mark.parametrize("streaming", [False, True])
def test_default_query_rewriter_uses_langchain_invoke(streaming):
    model = FakeListChatModel(responses=[
        '{"standalone_query":"RAG 的父子分块如何工作？"}',
        '先检索子块，再展开父块。[1]',
    ])
    service = QAService(model=model)
    router = RecordingRouter()
    service.retriever = router
    kwargs = {"history": [{"role": "user", "content": "我们在讨论 RAG 的父子分块"}]}
    if streaming:
        events = list(service.query_stream("它如何工作？", **kwargs))
        payload = events[-1][1]
        assert events[-1][0] == "done"
    else:
        payload = service.query("它如何工作？", **kwargs)
    assert router.questions == ["RAG 的父子分块如何工作？"]
    assert payload["trace"]["query_rewrite"]["applied"] is True
    assert payload["status"] == "answered"
    assert payload["citations"][0]["doc_id"] == "d"


def test_budget_does_not_hide_corrupt_child_coordinates():
    service = QAService()
    service.context_limit_tokens = 200
    service.completion_reserve_tokens = 10
    service.prompt_safety_tokens = 0
    chunk = Document(page_content="长文本 " * 1000, metadata={
        "doc_id": "d", "chunk_id": "p", "score": 0.9,
        "primary_matched_child_id": "c", "matched_children": [{
            "chunk_id": "c", "snippet": "损坏的坐标",
            "parent_char_start": 100000, "parent_char_end": 100010,
        }],
    })
    with pytest.raises(ValueError, match="outside"):
        service._budgeted_evidence("问题", [chunk], QAConfig(evidence_mode="full_parent"))
