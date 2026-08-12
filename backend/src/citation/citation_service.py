import re

from backend.src.citation.models import Citation, validate_citation
from backend.src.retrieval.models import RetrievedChunk


class CitationService:
    """Answer-evidence binding service."""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def build(self, answer: str, chunks: list[RetrievedChunk]) -> dict:
        referenced_indexes: list[int] = []
        invalid_indexes: list[int] = []
        seen: set[int] = set()
        for match in self._CITATION_PATTERN.finditer(answer):
            citation_index = int(match.group(1))
            if citation_index in seen:
                continue
            seen.add(citation_index)
            if 1 <= citation_index <= len(chunks):
                referenced_indexes.append(citation_index)
            else:
                invalid_indexes.append(citation_index)

        citations: list[Citation] = []
        cited_chunks: list[RetrievedChunk] = []
        for citation_index in referenced_indexes:
            chunk = chunks[citation_index - 1]
            snippet = " ".join(chunk.content.split())
            if len(snippet) > 160:
                snippet = f"{snippet[:160].rstrip()}..."
            citation = Citation(
                citation_index=citation_index,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                snippet=snippet,
                page_no=chunk.page_no,
                section_path=chunk.section_path,
                context_chunk_id=chunk.chunk_id,
                context_snippet=snippet,
                context_span=dict(chunk.context_span),
                prompt_span=dict(chunk.prompt_span),
                matched_children=list(chunk.matched_children),
                primary_matched_child_id=(
                    chunk.primary_matched_child_id or chunk.matched_child_id
                ),
                source_span=dict(chunk.source_span),
            )
            validate_citation(citation)
            citations.append(citation)
            cited_chunks.append(chunk)

        avg_score = 0.0
        if cited_chunks:
            avg_score = round(
                sum(chunk.score for chunk in cited_chunks) / len(cited_chunks),
                4,
            )

        return {
            "answer": answer,
            "citations": [citation.__dict__ for citation in citations],
            "trace": {
                "citation_count": len(citations),
                "evidence_count": len(chunks),
                "referenced_indexes": referenced_indexes,
                "invalid_indexes": invalid_indexes,
                "citation_avg_score": avg_score,
            },
        }
