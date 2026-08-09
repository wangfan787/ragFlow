"""引用阶段的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    citation_index: int
    chunk_id: str
    doc_id: str
    snippet: str
    page_no: int | None
    section_path: list[str]


def validate_citation(citation: Citation) -> None:
    if citation.citation_index <= 0:
        raise ValueError("citation_index must be > 0")
    if not citation.doc_id or not citation.chunk_id:
        raise ValueError("citation must include doc_id and chunk_id")
    if not citation.snippet.strip():
        raise ValueError("citation snippet cannot be empty")