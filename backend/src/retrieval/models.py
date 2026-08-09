"""检索阶段的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    content: str
    score: float
    vector_score: float
    keyword_score: float
    fused_score: float
    rerank_score: float | None
    section_path: list[str]
    page_no: int | None
    doc_name: str = ""
    important_kwd: list[str] = field(default_factory=list)
    question_kwd: list[str] = field(default_factory=list)
    embedding_backend: str = ""
    embedding_model: str = ""
    embedding_dim: int | None = None
    # 父子分块字段（来自 chunker 写入，经检索 payload 透传）：
    chunk_role: str = "parent"
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    chunk_order: int | None = None
    matched_child_id: str | None = None


def validate_retrieved_chunk(chunk: RetrievedChunk) -> None:
    if not chunk.doc_id or not chunk.chunk_id:
        raise ValueError("doc_id and chunk_id are required")
    for score in (chunk.score, chunk.vector_score, chunk.keyword_score, chunk.fused_score):
        if not 0.0 <= score <= 1.0:
            raise ValueError("scores must be between 0 and 1")
    if chunk.rerank_score is not None and not 0.0 <= chunk.rerank_score <= 1.0:
        raise ValueError("scores must be between 0 and 1")