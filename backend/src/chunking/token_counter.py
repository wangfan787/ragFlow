from __future__ import annotations

import os
import re
from typing import Protocol

import tiktoken
_API_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", _API_DIR)
_encoder = tiktoken.get_encoding("cl100k_base")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class SimpleTokenCounter:
    """基于 tiktoken cl100k_base 的 token 计数（与 RAGFlow 实现一致）。

    英文按 BPE 子词、中文按词表切分，计数口径与 OpenAI 模型一致，供分块阈值
    （parent/child target_tokens）使用。
    """

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0

        try:
            return len(_encoder.encode(text, disallowed_special=()))
        except Exception:
            return 0


    def tokens(self, text: str) -> list[str]:

        try:
            return [
                _encoder.decode_single_token_bytes(t).decode("utf-8", errors="replace")
                for t in _encoder.encode(text, disallowed_special=())
            ]
        except Exception:
            return []

    def truncate(self, text: str, max_tokens: int) -> str:
        """按 cl100k_base token 边界截断文本。"""
        if max_tokens <= 0:
            return ""
        try:
            token_ids = _encoder.encode(text, disallowed_special=())
            if len(token_ids) <= max_tokens:
                return text
            return _encoder.decode(token_ids[:max_tokens])
        except Exception as exc:
            raise RuntimeError("failed to tokenize embedding input") from exc



# ============================================================================
# 演示和测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Token Counter 演示")
    print("=" * 60)

    counter = SimpleTokenCounter()

    print("=" * 60)

    # 示例 1: 基本使用
    print("\n📝 示例 1: 基本使用")
    print("-" * 40)
    samples = [
        ("Hello world", "简单英文"),
        ("你好世界", "纯中文"),
        ("machine learning", "英文复合词"),
        ("", "空字符串"),
        ("   ", "纯空格"),
    ]

    for text, desc in samples:
        count = counter.count(text)
        print(f"{desc:12s}: '{text}' → {count} tokens")

    # 示例 2: tokens() 方法 - 查看分词结果
    print("\n🔍 示例 2: tokens() 方法 - 查看分词结果")
    print("-" * 40)
    debug_samples = [
        "Hello world",
        "machine learning",
        "你好世界",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in debug_samples:
        tokens = counter.tokens(text)
        print(f"原文: '{text}'")
        print(f"Token 数: {counter.count(text)}")
        print(f"分词结果: {tokens}")
        print()
