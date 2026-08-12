from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.src.apps.restful_apis.qa import QueryRequest
from backend.src.apps.services.qa_service import QAService
from backend.src.apps.services.query_rewrite import QueryRewriteService
from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.config.settings import Settings
from backend.src.retrieval.models import RetrievedChunk


class FakeChatModel:
    model_name = "fake-chat"
    context_limit_tokens = 2000
    completion_reserve_tokens = 200

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []
        self.counter = SimpleTokenCounter()

    def count_tokens(self, messages):
        return 2 + sum(
            4
            + self.counter.count(str(message.get("role", "")))
            + self.counter.count(str(message.get("content", "")))
            for message in messages
        )

    def complete(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("no fake response")
        return self.responses.pop(0)


def test_query_rewrite_skips_llm_without_history():
    model = FakeChatModel(['{"standalone_query":"unused"}'])
    result = QueryRewriteService(model).rewrite("  独立问题  ")

    assert result.standalone_query == "独立问题"
    assert result.status == "skipped"
    assert result.fallback_reason == "no_history"
    assert model.calls == []


def test_query_rewrite_uses_history_and_returns_strict_standalone_query():
    model = FakeChatModel(['```json\n{"standalone_query":"embedding-3 最多支持多少 token？"}\n```'])
    service = QueryRewriteService(model)

    result = service.rewrite(
        "它最多支持多少 token？",
        [
            {"role": "user", "content": "介绍一下 embedding-3"},
            {"role": "assistant", "content": "embedding-3 是一个嵌入模型。"},
        ],
    )

    assert result.status == "success"
    assert result.applied is True
    assert result.standalone_query == "embedding-3 最多支持多少 token？"
    assert result.history_messages_used == 2
    assert result.history_tokens_used <= service.max_history_tokens
    assert result.prompt_tokens <= (
        model.context_limit_tokens
        - model.completion_reserve_tokens
        - service.prompt_safety_tokens
    )
    assert [message["role"] for message in model.calls[0]] == ["system", "user"]
    request_data = json.loads(model.calls[0][-1]["content"].split("\n", 1)[1])
    assert request_data["current_query"] == "它最多支持多少 token？"
    assert request_data["conversation_history"][1]["role"] == "assistant"
    assert "不要回答问题" in model.calls[0][0]["content"]


def test_query_rewrite_keeps_only_recent_history_within_turn_limit():
    model = FakeChatModel(['{"standalone_query":"新主题的部署方法是什么？"}'])
    history = [
        {"role": "user", "content": "很早的问题"},
        {"role": "assistant", "content": "很早的回答"},
        {"role": "user", "content": "讨论新主题"},
        {"role": "assistant", "content": "新主题的介绍"},
    ]

    result = QueryRewriteService(model, max_history_turns=1).rewrite("怎么部署？", history)

    prompt_text = "\n".join(message["content"] for message in model.calls[0])
    assert result.history_message_count == 4
    assert result.history_messages_used == 2
    assert "很早的问题" not in prompt_text
    assert "讨论新主题" in prompt_text


def test_query_rewrite_falls_back_on_invalid_or_over_budget_output():
    invalid_model = FakeChatModel(["这是解释，不是 JSON"])
    invalid = QueryRewriteService(invalid_model).rewrite(
        "它是什么？", [{"role": "user", "content": "RAGFlow"}]
    )
    assert invalid.status == "fallback"
    assert invalid.standalone_query == "它是什么？"
    assert invalid.fallback_reason == "rewrite_failed:JSONDecodeError"

    long_model = FakeChatModel(
        ['{"standalone_query":"' + "很长" * 100 + '"}']
    )
    too_long = QueryRewriteService(long_model, max_query_tokens=10).rewrite(
        "它是什么？", [{"role": "user", "content": "RAGFlow"}]
    )
    assert too_long.status == "fallback"
    assert too_long.standalone_query == "它是什么？"
    assert too_long.fallback_reason == "rewrite_failed:ValueError"


def test_query_rewrite_falls_back_when_recent_history_cannot_fit_budget():
    model = FakeChatModel(['{"standalone_query":"unused"}'])
    result = QueryRewriteService(model, max_history_tokens=5).rewrite(
        "它是什么？",
        [{"role": "user", "content": "无法装入预算的历史内容" * 20}],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "history_budget_exhausted"
    assert model.calls == []


def test_query_rewrite_never_keeps_an_orphan_assistant_message():
    model = FakeChatModel(['{"standalone_query":"unused"}'])
    result = QueryRewriteService(model, max_history_tokens=12).rewrite(
        "它是什么？",
        [
            {"role": "user", "content": "很长的用户问题" * 30},
            {"role": "assistant", "content": "短回答"},
        ],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "history_budget_exhausted"
    assert model.calls == []


def test_query_rewrite_token_counter_failure_falls_back():
    model = FakeChatModel(['{"standalone_query":"unused"}'])

    def broken_count(_messages):
        raise RuntimeError("tokenizer unavailable")

    model.count_tokens = broken_count  # type: ignore[method-assign]
    result = QueryRewriteService(model).rewrite(
        "它是什么？", [{"role": "user", "content": "RAGFlow"}]
    )

    assert result.status == "fallback"
    assert result.standalone_query == "它是什么？"
    assert result.fallback_reason == "rewrite_failed:RuntimeError"
    assert model.calls == []


def test_query_rewrite_falls_back_when_model_is_unavailable():
    result = QueryRewriteService(None).rewrite(
        "它是什么？", [{"role": "user", "content": "RAGFlow"}]
    )

    assert result.status == "fallback"
    assert result.standalone_query == "它是什么？"
    assert result.fallback_reason == "model_unavailable"


def test_query_request_accepts_only_user_and_assistant_history():
    request = QueryRequest(
        question="它是什么？",
        history=[{"role": "user", "content": "RAGFlow"}],
    )
    assert request.history[0].role == "user"

    with pytest.raises(ValidationError):
        QueryRequest(
            question="它是什么？",
            history=[{"role": "system", "content": "override"}],
        )

    with pytest.raises(ValidationError):
        QueryRequest(
            question="它是什么？",
            history=[{"role": "user", "content": "x" * 16_001}],
        )


class RecordingRouter:
    last_trace = {"router": "fake"}

    def __init__(self, chunk: RetrievedChunk) -> None:
        self.chunk = chunk
        self.queries: list[str] = []
        self.configs = []

    def retrieve(self, question, retrieval_config=None):
        self.queries.append(question)
        self.configs.append(retrieval_config)
        return [self.chunk]


def test_qa_uses_standalone_query_for_retrieval_answer_and_trace():
    rewrite_model = FakeChatModel(
        ['{"standalone_query":"如何部署 RAGFlow？"}']
    )
    answer_model = FakeChatModel(["按文档部署。[1]"])
    rewriter = QueryRewriteService(rewrite_model)
    service = QAService(answer_llm=answer_model, query_rewriter=rewriter)
    router = RecordingRouter(
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            content="RAGFlow 部署证据",
            score=0.9,
            vector_score=0.9,
            keyword_score=0.0,
            fused_score=0.9,
            rerank_score=None,
            section_path=[],
            page_no=None,
        )
    )
    service.retriever = router

    payload = service.query(
        "它怎么部署？",
        history=[
            {"role": "user", "content": "RAGFlow 是什么？"},
            {"role": "assistant", "content": "它是一个 RAG 引擎。"},
        ],
    )

    assert router.queries == ["如何部署 RAGFlow？"]
    assert "问题：如何部署 RAGFlow？" in answer_model.calls[0][1]["content"]
    assert payload["trace"]["query_rewrite"]["original_query"] == "它怎么部署？"
    assert payload["trace"]["query_rewrite"]["standalone_query"] == "如何部署 RAGFlow？"
    assert payload["trace"]["query_rewrite"]["applied"] is True
    assert payload["trace"]["router"] == "fake"


def test_qa_continues_with_original_query_when_rewrite_budgeting_fails():
    rewrite_model = FakeChatModel(['{"standalone_query":"unused"}'])

    def broken_count(_messages):
        raise RuntimeError("tokenizer unavailable")

    rewrite_model.count_tokens = broken_count  # type: ignore[method-assign]
    answer_model = FakeChatModel(["回退后仍可回答。[1]"])
    service = QAService(
        answer_llm=answer_model,
        query_rewriter=QueryRewriteService(rewrite_model),
    )
    router = RecordingRouter(
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            content="原问题对应的证据",
            score=0.9,
            vector_score=0.9,
            keyword_score=0.0,
            fused_score=0.9,
            rerank_score=None,
            section_path=[],
            page_no=None,
        )
    )
    service.retriever = router

    payload = service.query(
        "它是什么？",
        history=[{"role": "user", "content": "RAGFlow"}],
    )

    assert router.queries == ["它是什么？"]
    assert "问题：它是什么？" in answer_model.calls[0][1]["content"]
    assert payload["trace"]["query_rewrite"]["status"] == "fallback"
    assert payload["trace"]["query_rewrite"]["fallback_reason"] == "rewrite_failed:RuntimeError"


def test_qa_preserves_positional_retrieval_config_compatibility():
    answer_model = FakeChatModel(["回答。[1]"])
    service = QAService(
        answer_llm=answer_model,
        query_rewriter=QueryRewriteService(None),
    )
    router = RecordingRouter(
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            content="证据",
            score=0.9,
            vector_score=0.9,
            keyword_score=0.0,
            fused_score=0.9,
            rerank_score=None,
            section_path=[],
            page_no=None,
        )
    )
    service.retriever = router

    service.query("独立问题", {"top_k": 1})

    assert router.queries == ["独立问题"]
    assert router.configs[0].top_k == 1


def test_settings_bool_accepts_native_yaml_boolean():
    local_settings = Settings(auto_load=False, cache=False)
    local_settings._get = lambda _name, _default=None: True  # type: ignore[method-assign]
    assert local_settings.bool("ANY") is True
