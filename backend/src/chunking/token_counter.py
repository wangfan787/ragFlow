"""统一工程 token 估算：cl100k_base 不是 GLM 官方 tokenizer。"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import tiktoken


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    # 词表冷缓存时可能下载；延迟到首次计数，应用导入和健康检查不联网。
    # 保留 tiktoken 自身的缓存目录规则和显式 TIKTOKEN_CACHE_DIR 设置。
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """按原始字符串计数（包含空白），编码失败直接报错，不能伪报为 0。"""
    if not text:
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def count_message_tokens(messages: list[dict]) -> int:
    """估算标准文本消息及消息封装；供应商差异由 QA 安全余量覆盖。"""
    return 2 + sum(4 + count_tokens(m["role"]) + count_tokens(m["content"]) for m in messages)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class SimpleTokenCounter:
    """旧消费者的薄适配；与新 count_tokens 使用同一个计数入口。"""

    def count(self, text: str) -> int:
        return count_tokens(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        """旧接口的截断保护，不作为新入库链路的长文本切片算法。"""
        if not text or max_tokens <= 0:
            return ""
        encoder = _get_encoder()
        token_ids = encoder.encode(text, disallowed_special=())
        if len(token_ids) <= max_tokens:
            return text
        # 一个中文字符可能跨多个 token；忽略不完整 UTF-8 尾部，避免引入 �。
        end = max_tokens
        while end > 0:
            prefix = encoder.decode(token_ids[:end], errors="ignore")
            if count_tokens(prefix) <= max_tokens:
                return prefix
            end -= 1
        return ""
