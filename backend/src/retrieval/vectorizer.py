
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.]+|[\u4e00-\u9fff]+", re.UNICODE)
_CJK_STOP_CHARS = {
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "是",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "为",
    "请",
    "可",
    "以",
    "和",
    "及",
    "在",
    "对",
    "给",
    "把",
    "个",
    "一",
    "下",
    "这",
    "那",
    "中",
}
_QUERY_STOP_TERMS = {
    "介",
    "绍",
    "介绍",
    "讲",
    "说",
    "说明",
    "请",
    "请问",
    "一下",
    "可以",
    "能",
    "不能",
    "能否",
    "帮",
    "帮我",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            chars = [char for char in token if char not in _CJK_STOP_CHARS]
            tokens.extend(chars)
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
        else:
            normalized = token.strip("._")
            if normalized:
                tokens.append(normalized)
    return tokens


def query_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if term not in _QUERY_STOP_TERMS}
