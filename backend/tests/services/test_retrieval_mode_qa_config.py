"""算法层 §4.2 前置改造的验收测试：

1. retrieval_mode 严格通道门控（未选中的通道不得执行）
2. rerank_top_n 固定"召回池 -> 送排池 -> top_k"漏斗
3. 请求级 QAConfig（context_top_k / evidence_mode / evidence_window_tokens）
4. 请求级结构化 trace（config / timings_ms / usage）
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.apps.services.qa_service import QAService
from backend.src.config.qa_config import QAConfig, build_qa_config
from backend.src.config.retrieval_config import build_retrieval_config
from backend.src.retrieval.hybrid_router import HybridRouter


class FakeEmbedding:
    model = "fake-v1"
    dimensions = 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, records=None):
        self.records = records or []

    def query_by_ids(self, ids):
        return [record for record in self.records if record.metadata["chunk_id"] in ids]

    def vector_search(self, query_vector, top_k, filters=None):
        return []

    def keyword_search(self, query, top_k, filters=None):
        return []


def _child_row(chunk_id: str, score: float, *, keyword: bool = False, parent_id: str | None = None) -> dict:
    """构造一个通道召回的 Child 行（与真实 Retriever 的输出结构一致）"""
    row = {
        "doc_id": "d", "chunk_id": chunk_id, "content": f"正文-{chunk_id}",
        "chunk_role": "child", "retrieval_eligible": True,
        "parent_id": parent_id or f"p-{chunk_id}",
        "vector_score": score if not keyword else 0.0,
        "keyword_score": score if keyword else 0.0,
        "fused_score": score,
        "score": score,
    }
    return row


def _gated_router(vector_rows=None, keyword_rows=None) -> tuple[HybridRouter, dict]:
    """构造可计数的 HybridRouter：记录每个通道被调用的次数"""
    router = HybridRouter(store=FakeStore(), embedding_model=FakeEmbedding())
    calls = {"vector": 0, "keyword": 0}

    def vector_retrieve(query, config=None):
        calls["vector"] += 1
        return list(vector_rows or [])

    def keyword_retrieve(query, config=None):
        calls["keyword"] += 1
        return list(keyword_rows or [])

    router.embedding_retriever.retrieve = vector_retrieve
    router.keyword_retriever.retrieve = keyword_retrieve
    return router, calls


# ---------------------------------------------------------------------------
# 1. retrieval_mode 通道门控
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode,expected",
    [
        ("vector", {"vector": 1, "keyword": 0}),
        ("keyword", {"vector": 0, "keyword": 1}),
        ("hybrid", {"vector": 1, "keyword": 1}),
    ],
)
def test_retrieval_mode_gates_unselected_channels(mode, expected):
    # Q24：weight=0 不等于不执行；只有 retrieval_mode 能保证通道不执行
    router, calls = _gated_router([_child_row("c1", 0.9)], [_child_row("c1", 0.5, keyword=True)])
    results, trace = router.retrieve_detailed("问题", {"retrieval_mode": mode})
    assert calls == expected
    assert trace["retrieval_mode"] == mode
    assert trace["vector_candidate_count"] == (1 if mode in ("vector", "hybrid") else 0)
    assert trace["keyword_candidate_count"] == (1 if mode in ("keyword", "hybrid") else 0)


def test_vector_weight_one_is_not_vector_only():
    # 权重全部给向量时，BM25 仍然执行——严格消融必须使用 retrieval_mode
    router, calls = _gated_router([_child_row("c1", 0.9)], [])
    router.retrieve_detailed("问题", {"retrieval_mode": "hybrid", "vector_weight": 1.0})
    assert calls == {"vector": 1, "keyword": 1}


def test_single_channel_mode_keeps_raw_channel_score_scale():
    # 单通道模式下融合分应等于通道原始分（阈值才可跨模式比较）
    vector_router, _ = _gated_router([_child_row("c1", 0.6)], [])
    results, trace = vector_router.retrieve_detailed("问题", {"retrieval_mode": "vector"})
    assert results[0].metadata["score"] == pytest.approx(0.6)
    assert trace["vector_weight_effective"] == 1.0
    assert trace["keyword_weight_effective"] == 0.0

    keyword_router, _ = _gated_router([], [_child_row("c1", 0.4, keyword=True)])
    results, trace = keyword_router.retrieve_detailed("问题", {"retrieval_mode": "keyword"})
    assert results[0].metadata["score"] == pytest.approx(0.4)


def test_retrieval_mode_validation_and_normalization():
    config = build_retrieval_config(retrieval_mode=" Vector ")
    assert config.retrieval_mode == "vector"
    with pytest.raises(ValueError, match="retrieval_mode"):
        build_retrieval_config(retrieval_mode="semantic")
    with pytest.raises(ValueError, match="unsupported retrieval_config fields"):
        build_retrieval_config(unknown_field=1)


# ---------------------------------------------------------------------------
# 2. rerank_top_n 漏斗
# ---------------------------------------------------------------------------

class RecordingReranker:
    backend_name = "recording"

    def __init__(self):
        self.pool_sizes: list[int] = []

    def rerank(self, query, chunks):
        self.pool_sizes.append(len(chunks))
        return list(chunks)


def test_rerank_top_n_limits_pool_sent_to_reranker():
    router, _ = _gated_router([_child_row(f"c{i}", 0.9 - i * 0.1) for i in range(5)], [])
    reranker = RecordingReranker()
    router.reranker = reranker
    results, trace = router.retrieve_detailed(
        "问题",
        {
            "retrieval_mode": "vector",
            "rerank_enabled": True,
            "rerank_backend": "rule",
            "candidate_top_k": 10,
            "rerank_top_n": 3,
            "top_k": 2,
        },
    )
    # 5 个融合候选中只有前 rerank_top_n=3 个被送去重排
    assert reranker.pool_sizes == [3]
    assert len(results) == 2
    assert trace["rerank_sent_count"] == 3
    assert trace["rerank_top_n_effective"] == 3
    assert trace["candidate_top_k"] == 10
    assert trace["final_top_k"] == 2
    assert trace["rerank_backend"] == "rule"


def test_rerank_top_n_defaults_to_whole_fusion_pool():
    router, _ = _gated_router([_child_row(f"c{i}", 0.9 - i * 0.1) for i in range(4)], [])
    reranker = RecordingReranker()
    router.reranker = reranker
    _, trace = router.retrieve_detailed(
        "问题",
        {"retrieval_mode": "vector", "rerank_enabled": True, "candidate_top_k": 10},
    )
    # 未配置 rerank_top_n 时保持旧行为：重排整个融合池
    assert reranker.pool_sizes == [4]
    assert trace["rerank_top_n_effective"] == 10
    assert trace["rerank_sent_count"] == 4


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


def test_rerank_top_n_none_restores_default_semantics():
    config = build_retrieval_config({"rerank_top_n": None})
    assert config.rerank_top_n is None


# ---------------------------------------------------------------------------
# 3. 请求级 QAConfig
# ---------------------------------------------------------------------------

def test_qa_config_build_and_validation():
    config = build_qa_config({"evidence_mode": "Child_Only "}, evidence_window_tokens=192)
    assert config == QAConfig(context_top_k=5, evidence_mode="child_only", evidence_window_tokens=192)
    with pytest.raises(ValueError, match="evidence_mode"):
        build_qa_config(evidence_mode="everything")
    with pytest.raises(ValueError, match="unsupported qa_config fields"):
        build_qa_config(window_tokens=192)
    with pytest.raises(ValueError, match="cannot be None"):
        build_qa_config(evidence_window_tokens=None)
    with pytest.raises(ValueError):
        QAConfig(context_top_k=0)
    with pytest.raises(ValueError):
        QAConfig(evidence_window_tokens=0)


def _parent_chunk(text: str, child_snippet: str, child_span: tuple[int, int]) -> Document:
    start, end = child_span
    return Document(
        page_content=text,
        metadata={
            "chunk_id": "parent", "doc_id": "doc", "score": 0.8,
            "vector_score": 0.8, "keyword_score": 0.0, "fused_score": 0.8,
            "rerank_score": None, "section_path": ["章节"], "page_no": None,
            "chunk_role": "parent",
            "matched_child_id": "child", "primary_matched_child_id": "child",
            "matched_children": [{
                "chunk_id": "child", "score": 0.8, "snippet": child_snippet,
                "parent_char_start": start, "parent_char_end": end,
                "source_span": {"start_char": start, "end_char": end, "accuracy": "exact"},
            }],
        },
    )


class FakeDetailedRouter:
    """带请求级 trace 的检索替身：验证 QAService 不再依赖共享 last_trace。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.received_configs = []

    def retrieve_detailed(self, question, retrieval_config=None):
        self.received_configs.append(retrieval_config)
        return list(self.chunks), {
            "request_local": True,
            "retrieval_mode": retrieval_config.retrieval_mode,
        }


class FakeAnswerModel:
    model_name = "fake-qa-model"

    def __init__(self, content="答案 [1]", usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(
            content=self.content,
            usage_metadata=self.usage_metadata,
            response_metadata=None,
        )


def _qa_service(chunk: Document, model: FakeAnswerModel | None = None) -> tuple[QAService, FakeAnswerModel]:
    model = model or FakeAnswerModel()
    service = QAService(model=model)
    service.retriever = FakeDetailedRouter([chunk])
    return service, model


def test_qa_child_only_mode_cites_matched_child_text():
    text = "前文" * 100 + "命中子块内容" + "后文" * 100
    start, end = text.index("命中子块内容"), text.index("命中子块内容") + len("命中子块内容")
    chunk = _parent_chunk(text, "命中子块内容", (start, end))
    service, model = _qa_service(chunk)

    payload = service.query("问题", qa_config={"evidence_mode": "child_only"})

    # 证据只包含命中子块文本，引用指向子块而不是父块
    assert "命中子块内容" in model.messages[1]["content"]
    assert "前文前文" not in model.messages[1]["content"]
    assert payload["citations"][0]["chunk_id"] == "child"
    assert payload["trace"]["config"]["qa_config"]["evidence_mode"] == "child_only"


def test_qa_full_parent_mode_uses_entire_parent_text():
    text = "头部证据" + "中段" * 50 + "尾部证据"
    start = text.index("头部证据")
    chunk = _parent_chunk(text, "命中", (start, start + 2))
    service, model = _qa_service(chunk)

    # 窗口设得很小也不影响 full_parent：初始候选就是整个父块
    payload = service.query("问题", qa_config={"evidence_mode": "full_parent", "evidence_window_tokens": 16})

    assert "头部证据" in model.messages[1]["content"]
    assert "尾部证据" in model.messages[1]["content"]
    assert payload["trace"]["config"]["qa_config"]["evidence_mode"] == "full_parent"


def test_qa_window_mode_default_unchanged_and_request_scoped():
    text = "前文" * 2000 + "真正命中的尾部证据" + "后文" * 200
    start = text.index("真正命中的尾部证据")
    chunk = _parent_chunk(text, "真正命中的尾部证据", (start, start + len("真正命中的尾部证据")))
    service, model = _qa_service(chunk)

    # 同一服务实例连续两次请求使用不同窗口配置：请求级 QAConfig 生效
    service.query("第一问", qa_config={"evidence_window_tokens": 300})
    payload_two = service.query("第二问", qa_config={"evidence_window_tokens": 100000})

    assert payload_two["trace"]["config"]["qa_config"] == {
        "context_top_k": 5, "evidence_mode": "window", "evidence_window_tokens": 100000,
    }
    # 两次请求的 trace 都是请求局部对象
    assert payload_two["trace"]["request_local"] is True
    assert payload_two["trace"]["retrieval_mode"] == "hybrid"


def test_qa_service_passes_retrieval_config_to_router():
    chunk = _parent_chunk("短文本", "短", (0, 1))
    service, _ = _qa_service(chunk)
    service.query("问题", retrieval_config={"retrieval_mode": "vector", "top_k": 3})
    received = service.retriever.received_configs[0]
    assert received.retrieval_mode == "vector"
    assert received.top_k == 3


# ---------------------------------------------------------------------------
# 4. 结构化 trace：timings_ms / usage / config
# ---------------------------------------------------------------------------

def test_qa_trace_has_stage_timings_and_config_snapshot():
    chunk = _parent_chunk("用于计时的普通文本。", "用于", (0, 2))
    service, _ = _qa_service(chunk)

    payload = service.query("问题", retrieval_config={"retrieval_mode": "vector"}, qa_config={"context_top_k": 2})

    timings = payload["trace"]["timings_ms"]
    for stage in ("retrieval", "window", "generation", "citation", "total"):
        assert stage in timings and timings[stage] >= 0.0
    assert timings["total"] >= timings["retrieval"] + timings["window"] + timings["generation"]

    config_snapshot = payload["trace"]["config"]
    assert config_snapshot["retrieval_config"]["retrieval_mode"] == "vector"
    assert config_snapshot["qa_config"]["context_top_k"] == 2
    assert config_snapshot["model"] == "fake-qa-model"
    assert config_snapshot["index_name"]


def test_qa_usage_records_provider_numbers_when_available():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    model = FakeAnswerModel(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    service, _ = _qa_service(chunk, model)

    payload = service.query("问题")
    usage = payload["trace"]["usage"]
    assert usage == {
        "model": "fake-qa-model",
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "cost": "unavailable",
    }


def test_qa_usage_marks_missing_fields_unavailable_instead_of_zero():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    service, _ = _qa_service(chunk)  # FakeAnswerModel 不带 usage_metadata

    payload = service.query("问题")
    usage = payload["trace"]["usage"]
    # 供应商未返回 usage 时不得用 0 冒充真实消耗
    assert usage["input_tokens"] == "unavailable"
    assert usage["output_tokens"] == "unavailable"
    assert usage["total_tokens"] == "unavailable"
    assert usage["cost"] == "unavailable"


def test_qa_usage_reads_openai_compatible_token_usage():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    model = FakeAnswerModel()
    model.invoke = lambda messages: SimpleNamespace(
        content="答案 [1]",
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    service, _ = _qa_service(chunk, model)

    usage = service.query("问题")["trace"]["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
