from __future__ import annotations

import json

from backend.src.apps.services.query_rewrite import QueryRewriteService
from backend.src.chunking.token_counter import SimpleTokenCounter


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


