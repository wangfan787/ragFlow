from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from backend.src.chunking.token_counter import SimpleTokenCounter

logger = logging.getLogger("mvp_api")


def thinking_extra_body(base_url: str | None, thinking: str, log_prefix: str) -> dict:
    thinking = thinking.lower()
    if not thinking and base_url and "bigmodel.cn" in base_url:
        thinking = "disabled"
    if thinking in {"enabled", "disabled"}:
        return {"thinking": {"type": thinking}}
    if thinking:
        logger.warning("%s.llm_invalid_thinking value=%r", log_prefix, thinking)
    return {}


class OpenAIChat:
    """Thin wrapper around OpenAI-compatible chat completion APIs."""

    backend_name = "openai-compatible"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int,
        temperature: float = 0.2,
        extra_body: dict | None = None,
        context_limit_tokens: int = 32768,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.context_limit_tokens = context_limit_tokens
        self.completion_reserve_tokens = max_tokens
        self._token_counter = SimpleTokenCounter()

    @property
    def model_name(self) -> str:
        return self.model

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        # Conservative OpenAI-compatible envelope estimate. The safety budget
        # configured by QA absorbs provider-specific serialization variance.
        return 2 + sum(
            4
            + self._token_counter.count(str(message.get("role", "")))
            + self._token_counter.count(str(message.get("content", "")))
            for message in messages
        )

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        if content.strip():
            return content.strip()

        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        raise ValueError(
            "empty LLM response "
            f"finish_reason={getattr(choice, 'finish_reason', None)} "
            f"completion_tokens={getattr(usage, 'completion_tokens', None)} "
            f"reasoning_tokens={reasoning_tokens}"
        )
