"""/qa/query 与 /qa/query/stream HTTP 端到端（替身检索 + 替身模型）。

- 未带 token → 401；
- 合法请求 → 200，信封 code=OK，answer/citations/trace 齐全，
  retrieval_config / qa_config 从 HTTP body 逐层透传到检索与窗口阶段；
- 非法 retrieval_config / qa_config 字段 → 422；
- 检索无命中 → 200 且 data.status="no_evidence"（产品化拒答，非业务错误）；
- 流式端点按 answer/citation/done 事件序列输出 SSE，空检索只发 done。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document


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
    """记录收到的请求级配置与问题文本，返回固定父块与请求局部 trace。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.received_configs = []
        self.received_questions = []

    def retrieve_detailed(self, question, retrieval_config=None):
        self.received_configs.append(retrieval_config)
        self.received_questions.append(question)
        return list(self.chunks), {
            "request_local": True,
            "retrieval_mode": retrieval_config.retrieval_mode,
        }


class FakeAnswerModel:
    model_name = "fake-qa-model"

    def invoke(self, messages):
        return SimpleNamespace(
            content="答案 [1]",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            response_metadata=None,
        )


class FakeStreamModel:
    """带 .stream 的替身：两个文本增量；流分块不返回 usage。"""

    model_name = "fake-stream-model"

    def stream(self, messages):
        yield SimpleNamespace(content="答案", usage_metadata=None)
        yield SimpleNamespace(content=" [1]", usage_metadata=None)


def _install_fakes(monkeypatch, chunks=None, model=None) -> FakeDetailedRouter:
    from backend.src.apps.restful_apis import qa as qa_api

    router = FakeDetailedRouter(chunks if chunks is not None else [_parent_chunk("前文证据。", "命中", (0, 2))])
    # 直改私有属性，避免 retriever property 懒加载真实 HybridRouter
    monkeypatch.setattr(qa_api.service, "_retriever", router)
    monkeypatch.setattr(qa_api.service, "model", model or FakeAnswerModel())
    return router


def _client_with_token() -> tuple[TestClient, str]:
    from backend.src.apps.services.common_service import create_access_token
    from backend.src.main import create_app

    return TestClient(create_app()), create_access_token("alice")


def test_query_requires_auth() -> None:
    client, _ = _client_with_token()
    response = client.post("/qa/query", json={"question": "问题"})
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_query_end_to_end_with_fakes(monkeypatch) -> None:
    router = _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={
            "question": "什么是父子分块？",
            "retrieval_config": {"retrieval_mode": "vector", "top_k": 3},
            "qa_config": {"evidence_mode": "child_only"},
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    body = response.json()
    assert body["code"] == "OK"

    data = body["data"]
    assert data["answer"] == "答案 [1]"
    assert data["citations"][0]["chunk_id"] == "child"

    # 请求级配置逐层透传：HTTP body → 服务层 → 检索
    received = router.received_configs[0]
    assert received.retrieval_mode == "vector"
    assert received.top_k == 3
    assert data["trace"]["config"]["qa_config"]["evidence_mode"] == "child_only"
    assert data["trace"]["config"]["model"] == "fake-qa-model"
    assert data["trace"]["config"]["index_name"]

    # usage 记录供应商真实数字；分阶段耗时齐全
    assert data["trace"]["usage"]["total_tokens"] == 10
    timings = data["trace"]["timings_ms"]
    assert {"retrieval", "window", "generation", "citation", "total"} <= set(timings)


def test_query_rejects_unknown_retrieval_config_field(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "问题", "retrieval_config": {"unknown_field": 1}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert "不合法" in response.json()["message"]


def test_query_rejects_invalid_qa_config_value(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "问题", "qa_config": {"evidence_mode": "everything"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert "不合法" in response.json()["message"]


def test_query_empty_question_returns_422(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post("/qa/query", json={"question": "  "}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_QUESTION"


def test_query_returns_no_evidence_product_response(monkeypatch) -> None:
    """检索无命中：200 + status=no_evidence，不再是业务错误；trace 仍可录制。"""
    _install_fakes(monkeypatch, chunks=[])
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "完全无关的问题"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "OK"

    data = body["data"]
    assert data["status"] == "no_evidence"
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["retrieved_chunks"] == []
    # 拒答也要可评测：配置快照与实际执行过的阶段耗时保留；未调用模型，usage 不可用
    assert data["trace"]["config"]["model"]
    assert "retrieval" in data["trace"]["timings_ms"]
    assert data["trace"]["usage"]["total_tokens"] == "unavailable"


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[len("data: "):] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def test_query_stream_emits_answer_citation_done(monkeypatch) -> None:
    _install_fakes(monkeypatch, model=FakeStreamModel())
    client, token = _client_with_token()

    response = client.post(
        "/qa/query/stream",
        json={"question": "什么是父子分块？"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["answer", "answer", "citation", "done"]

    assert events[0][1] == {"delta": "答案"}
    assert events[1][1] == {"delta": " [1]"}
    # 默认 window 模式：引用挂父块；citation 事件与 done 终态一致
    assert events[2][1]["citations"][0]["chunk_id"] == "parent"

    final = events[3][1]
    assert final["status"] == "answered"
    assert final["answer"] == "答案 [1]"
    assert final["citations"] == events[2][1]["citations"]
    assert final["trace"]["config"]["model"] == "fake-stream-model"
    # 流分块未返回 usage：如实记 unavailable，不补零
    assert final["trace"]["usage"]["total_tokens"] == "unavailable"
    assert {"retrieval", "window", "generation", "citation", "total"} <= set(final["trace"]["timings_ms"])


def test_query_stream_reports_missing_stream_instead_of_falling_back(monkeypatch) -> None:
    """不完整的模型契约应暴露错误，不能静默换成同步调用。"""
    _install_fakes(monkeypatch, model=FakeAnswerModel())
    client, token = _client_with_token()

    response = client.post(
        "/qa/query/stream",
        json={"question": "问题"},
        headers={"Authorization": f"Bearer {token}"},
    )
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["code"] == "GENERATION_FAILED"
    assert "stream" in events[0][1]["details"]["reason"]


def test_query_stream_no_evidence_emits_single_done(monkeypatch) -> None:
    _install_fakes(monkeypatch, chunks=[], model=FakeStreamModel())
    client, token = _client_with_token()

    response = client.post(
        "/qa/query/stream",
        json={"question": "完全无关的问题"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["done"]
    assert events[0][1]["status"] == "no_evidence"
    assert events[0][1]["answer"] == ""


def test_query_stream_rejects_invalid_config_before_stream(monkeypatch) -> None:
    """配置校验挡在流开始之前：返回 422 而不是半途断流。"""
    _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query/stream",
        json={"question": "问题", "retrieval_config": {"unknown_field": 1}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert "不合法" in response.json()["message"]


class FakeRewriteResult:
    """与 QueryRewriteResult 同构的最小替身：trace() 返回请求局部字典。"""

    def __init__(self, standalone_query: str, status: str = "success", applied: bool = True):
        self.original_query = "它多少钱？"
        self.standalone_query = standalone_query
        self.status = status
        self.applied = applied

    def trace(self) -> dict:
        return {
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "status": self.status,
            "applied": self.applied,
        }


class FakeQueryRewriter:
    """记录收到的 (question, history)，返回预置改写结果。"""

    def __init__(self, result: FakeRewriteResult):
        self.result = result
        self.calls: list[tuple[str, list]] = []

    def rewrite(self, question, history=None):
        self.calls.append((question, list(history or [])))
        return self.result


def _install_rewriter(monkeypatch, rewriter: FakeQueryRewriter) -> None:
    from backend.src.apps.restful_apis import qa as qa_api

    monkeypatch.setattr(qa_api.service, "_query_rewriter", rewriter)


_HISTORY = [
    {"role": "user", "content": "ChatLog 3.0 发布了吗？"},
    {"role": "assistant", "content": "发布了。"},
]


def test_query_applies_rewrite_with_history(monkeypatch) -> None:
    """带历史时执行改写：检索与后续链路使用独立问句，trace 记录改写结果。"""
    router = _install_fakes(monkeypatch)
    rewriter = FakeQueryRewriter(FakeRewriteResult("ChatLog 3.0 的价格是多少？"))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "它多少钱？", "history": _HISTORY},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    # 改写服务收到原始问题与历史；检索收到独立问句
    assert rewriter.calls == [("它多少钱？", _HISTORY)]
    assert router.received_questions == ["ChatLog 3.0 的价格是多少？"]
    assert data["answer"] == "答案 [1]"

    # 请求级开关进配置快照；改写阶段进 trace 与耗时
    assert data["trace"]["config"]["qa_config"]["query_rewrite_enabled"] is True
    rewrite_trace = data["trace"]["query_rewrite"]
    assert rewrite_trace["status"] == "success"
    assert rewrite_trace["applied"] is True
    assert rewrite_trace["standalone_query"] == "ChatLog 3.0 的价格是多少？"
    assert "query_rewrite" in data["trace"]["timings_ms"]


def test_query_rewrite_fallback_keeps_original_question(monkeypatch) -> None:
    """改写失败回退原问题：链路继续，trace 如实记录 fallback。"""
    router = _install_fakes(monkeypatch)
    rewriter = FakeQueryRewriter(FakeRewriteResult("它多少钱？", status="fallback", applied=False))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "它多少钱？", "history": _HISTORY},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    assert router.received_questions == ["它多少钱？"]
    assert data["trace"]["query_rewrite"]["status"] == "fallback"
    assert data["trace"]["query_rewrite"]["applied"] is False


def test_query_without_history_skips_rewrite_stage(monkeypatch) -> None:
    """无历史不执行改写：不碰改写服务，trace/timings 不补零。"""
    router = _install_fakes(monkeypatch)
    rewriter = FakeQueryRewriter(FakeRewriteResult("不应被调用"))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post("/qa/query", json={"question": "独立问题"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()["data"]

    assert rewriter.calls == []
    assert router.received_questions == ["独立问题"]
    assert "query_rewrite" not in data["trace"]
    assert "query_rewrite" not in data["trace"]["timings_ms"]


def test_query_rewrites_skipped_when_disabled_by_qa_config(monkeypatch) -> None:
    """请求级关闭：即使带历史也跳过改写阶段。"""
    router = _install_fakes(monkeypatch)
    rewriter = FakeQueryRewriter(FakeRewriteResult("不应被调用"))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={
            "question": "独立问题",
            "history": _HISTORY,
            "qa_config": {"query_rewrite_enabled": False},
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    assert rewriter.calls == []
    assert data["trace"]["config"]["qa_config"]["query_rewrite_enabled"] is False
    assert "query_rewrite" not in data["trace"]


def test_query_rejects_invalid_query_rewrite_flag(monkeypatch) -> None:
    """布尔开关严格解析：非 bool/"true"/"false" 取值返回 422。"""
    _install_fakes(monkeypatch)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "问题", "qa_config": {"query_rewrite_enabled": "yes"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert "不合法" in response.json()["message"]


def test_query_stream_applies_rewrite_and_records_trace(monkeypatch) -> None:
    """流式链路同样执行改写：done 终态 payload 携带 query_rewrite trace。"""
    router = _install_fakes(monkeypatch, model=FakeStreamModel())
    rewriter = FakeQueryRewriter(FakeRewriteResult("ChatLog 3.0 的价格是多少？"))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query/stream",
        json={"question": "它多少钱？", "history": _HISTORY},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[-1][0] == "done"

    final = events[-1][1]
    assert router.received_questions == ["ChatLog 3.0 的价格是多少？"]
    assert final["trace"]["query_rewrite"]["applied"] is True
    assert "query_rewrite" in final["trace"]["timings_ms"]


class FakeStepBackModel:
    """背景扩展替身：证据生成提示返回答案，背景扩展提示返回严格 JSON。"""

    model_name = "fake-stepback-model"

    def invoke(self, messages):
        user_content = messages[-1]["content"]
        if user_content.startswith("问题："):
            return SimpleNamespace(
                content="答案 [1]",
                usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                response_metadata=None,
            )
        return SimpleNamespace(
            content='{"step_back_query": "检索背景原理是什么？"}',
            usage_metadata=None,
            response_metadata=None,
        )


class FakeMultiQueryRouter:
    """按问题返回不同命中的替身：验证 Step-back 双路检索合并。"""

    def __init__(self):
        self.received_questions = []

    def retrieve_detailed(self, question, retrieval_config=None):
        self.received_questions.append(question)
        if "背景" in question:
            chunk = _parent_chunk("背景原理证据。", "背景命中", (0, 4))
            chunk.metadata["chunk_id"] = "parent-bg"
        else:
            chunk = _parent_chunk("前文证据。", "命中", (0, 2))
        return [chunk], {"request_local": True, "retrieval_mode": retrieval_config.retrieval_mode}


def test_query_step_back_merges_background_retrieval(monkeypatch) -> None:
    """Step-back 开启：背景问题单独检索后合并补漏，trace 记录扩展结果。"""
    from backend.src.apps.restful_apis import qa as qa_api

    router = FakeMultiQueryRouter()
    monkeypatch.setattr(qa_api.service, "_retriever", router)
    monkeypatch.setattr(qa_api.service, "model", FakeStepBackModel())
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={
            "question": "为什么这个锁在这里无效？",
            "qa_config": {"step_back_enabled": True},
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    # 原问题与背景问题各检索一次
    assert router.received_questions == [
        "为什么这个锁在这里无效？",
        "检索背景原理是什么？",
    ]
    step_back = data["trace"]["step_back"]
    assert step_back["status"] == "success"
    assert step_back["applied"] is True
    assert step_back["step_back_query"] == "检索背景原理是什么？"
    assert step_back["chunks_from_step_back"] == 1
    # 两路命中合并进结果；生成提示词仍基于原问题
    assert data["answer"] == "答案 [1]"
    assert {chunk["chunk_id"] for chunk in data["retrieved_chunks"]} == {"parent", "parent-bg"}
    assert {"step_back", "step_back_retrieval"} <= set(data["trace"]["timings_ms"])
    assert data["trace"]["config"]["qa_config"]["step_back_enabled"] is True


def test_query_step_back_fallback_keeps_single_retrieval(monkeypatch) -> None:
    """背景问题生成失败：回退单路检索，trace 如实记录 fallback。"""
    _install_fakes(monkeypatch, model=FakeAnswerModel())
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={"question": "问题", "qa_config": {"step_back_enabled": True}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    step_back = data["trace"]["step_back"]
    assert step_back["status"] == "fallback"
    assert step_back["fallback_reason"] == "step_back_failed:JSONDecodeError"
    assert step_back["applied"] is False
    assert "step_back_retrieval" not in data["trace"]["timings_ms"]
    assert "step_back" in data["trace"]["timings_ms"]
    assert data["answer"] == "答案 [1]"


def test_query_colloquial_normalization_without_history(monkeypatch) -> None:
    """开启口语规范化：无历史也执行改写，检索使用规范化问句。"""
    router = _install_fakes(monkeypatch)
    rewriter = FakeQueryRewriter(FakeRewriteResult("查询改写是如何实现的？"))
    _install_rewriter(monkeypatch, rewriter)
    client, token = _client_with_token()

    response = client.post(
        "/qa/query",
        json={
            "question": "查询改写咋实现的？",
            "qa_config": {"colloquial_normalization_enabled": True},
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()["data"]

    assert rewriter.calls == [("查询改写咋实现的？", [])]
    assert router.received_questions == ["查询改写是如何实现的？"]
    assert data["trace"]["query_rewrite"]["applied"] is True
    assert data["trace"]["config"]["qa_config"]["colloquial_normalization_enabled"] is True
