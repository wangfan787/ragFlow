"""引用阶段的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class Citation:
    citation_index: int
    chunk_id: str
    doc_id: str
    snippet: str
    page_no: int | None
    section_path: list[str]
    context_chunk_id: str = ""
    context_snippet: str = ""
    context_span: dict = field(default_factory=dict)
    prompt_span: dict = field(default_factory=dict)
    matched_children: list[dict] = field(default_factory=list)
    primary_matched_child_id: str | None = None
    source_span: dict = field(default_factory=dict)


def validate_citation(citation: Citation) -> None:
    if citation.citation_index <= 0:
        raise ValueError("citation_index must be > 0")
    if not citation.doc_id or not citation.chunk_id:
        raise ValueError("citation must include doc_id and chunk_id")
    if not citation.snippet.strip():
        raise ValueError("citation snippet cannot be empty")
    if citation.context_chunk_id != citation.chunk_id:
        raise ValueError("citation context_chunk_id must identify the prompt evidence")
