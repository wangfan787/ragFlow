from __future__ import annotations

from backend.src.contracts import Reranker
from backend.src.retrieval.ranking import chunk_order
from backend.src.retrieval.vectorizer import query_terms, tokenize


class RuleReranker(Reranker):
    backend_name = "rule"

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        terms = query_terms(query)
        reranked: list[dict] = []
        for item in chunks:
            text = " ".join(
                [
                    str(item.get("doc_name", "")),
                    " ".join(str(part) for part in item.get("section_path", [])),
                    " ".join(str(part) for part in item.get("important_kwd", [])),
                    " ".join(str(part) for part in item.get("important_tks", [])),
                    " ".join(str(part) for part in item.get("question_kwd", [])),
                    " ".join(str(part) for part in item.get("question_tks", [])),
                    str(item.get("content", "")),
                ]
            )
            text_terms = set(tokenize(text))
            direct_overlap = len(terms & text_terms)
            base_score = float(item.get("fused_score", item.get("score", 0.0)))
            coverage = direct_overlap / len(terms) if terms else 0.0
            rerank_score = min(1.0, base_score + 0.1 * coverage)
            reranked.append({**item, "rerank_score": rerank_score, "score": rerank_score})

        reranked.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("keyword_score", 0.0)),
                -chunk_order(row),
                float(row.get("vector_score", 0.0)),
            ),
            reverse=True,
        )
        return reranked
