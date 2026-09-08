"""md-v1 公共契约；文本统一用 LangChain Document，metadata 只做类型提示。

字符区间为父块规范化文本中的 [start, end)，行号从 1 开始且两端包含。
Document 已接入业务；下列 md-v1 字段提示随各阶段逐项落实。
"""

from __future__ import annotations

from typing import Literal, TypedDict

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


class SourceSpan(TypedDict):
    """原始结构块行范围可宽于展示片段，不承诺逐字来源坐标。"""

    start: int
    end: int
    line_start: int
    line_end: int
    asset_id: str | None


class MatchedChild(TypedDict):
    chunk_id: str
    score: float
    vector_score: float
    keyword_score: float
    parent_start: int
    parent_end: int


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
    role: Literal["parent", "child"]
    parent_id: str | None
    parent_start: int
    parent_end: int
    spans: list[SourceSpan]
    score: float
    matched_children: list[MatchedChild]
    evidence_id: int
    window_start: int
    window_end: int
