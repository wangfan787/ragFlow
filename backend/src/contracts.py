"""md-v1 公共契约；文本统一用 LangChain Document，metadata 只做类型提示。

字符区间为父块规范化文本中的 [start, end)，行号从 1 开始且两端包含。
Document 已接入业务；下列 md-v1 字段提示随各阶段逐项落实。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

from langchain_core.documents import Document

SCHEMA_VERSION = "md-v1"

class DocumentRecord(TypedDict):
    doc_id: str
    name: str
    source_path: str
    batch_id: str
    status: Literal["uploaded", "indexing", "ready", "failed"]
    error: str | None
    assets: dict[str, dict[str, str]]  # 每项 relative_path/mime_type/description
    counts: dict[str, int]  # blocks/parents/children/images


class SourceSpan(TypedDict, total=False):
    """来源跨度；键与 block_chunker._merge_span 的输出一致，不承诺逐字坐标。"""

    start_line: int | None
    end_line: int | None
    start_char: int | None
    end_char: int | None
    page_no: int | None
    accuracy: Literal["exact", "line_only", "unavailable"]


class MatchedChild(TypedDict, total=False):
    """与 hybrid_router 产出的 matched_children[] 字段一一对齐。"""

    chunk_id: str
    score: float
    vector_score: float
    keyword_score: float
    fused_score: float
    rerank_score: float | None
    snippet: str
    source_span: dict
    source_block_ids: list[str]
    parent_char_start: int | None
    parent_char_end: int | None


class DocumentMetadata(TypedDict, total=False):
    """各阶段共用字段提示；必填字段及坐标规则以 plan.md 为准。"""

    doc_id: str
    name: str
    source_path: str
    section_path: list[str]
    asset_ids: list[str]
    block_id: str
    kind: Literal["heading", "text", "code", "table", "image"]
    language: str | None
    line_start: int
    line_end: int
    chunk_id: str
    chunk_role: Literal["parent", "child"]
    parent_id: str | None
    parent_start: int
    parent_end: int
    spans: list[SourceSpan]
    score: float
    matched_children: list[MatchedChild]
    evidence_id: int
    window_start: int
    window_end: int


EmbeddingVector = list[float]


def validate_embedding_vector(values: Sequence[float]) -> None:
    """向量基础合法性；维度一致性与零向量等业务校验由 indexer 负责。"""

    if not values:
        raise ValueError("embedding vector must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("embedding vector contains a non-numeric value")
        if not math.isfinite(float(value)):
            raise ValueError("embedding vector contains a non-finite value")


class EmbeddingModel(Protocol):
    backend_name: str
    model_name: str
    dimensions: int | None
    max_input_tokens: int | None

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]: ...


class Reranker(Protocol):
    backend_name: str

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]: ...


class ChatModel(Protocol):
    """问答/改写模型能力协议；completion_reserve_tokens 才是 API 的 max_tokens 语义。"""

    model_name: str
    context_limit_tokens: int
    completion_reserve_tokens: int

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int: ...

    def complete(self, messages: Sequence[dict[str, str]]) -> str: ...
