from __future__ import annotations

from backend.src.retrieval.ranking import chunk_order


class HybridFusion:
    """Fuse results from keyword and vector channels."""

    def fuse(
        self,
        vector_chunks: list[dict],
        keyword_chunks: list[dict],
        *,
        vector_weight: float,
        keyword_weight: float,
    ) -> list[dict]:
        merged: dict[tuple[str, str], dict] = {}

        for item in vector_chunks:
            key = (str(item["doc_id"]), str(item["chunk_id"]))
            merged[key] = {
                **item,
                "vector_score": float(item.get("vector_score", item.get("score", 0.0))),
                "keyword_score": float(item.get("keyword_score", 0.0)),
                "channel_hits": list(item.get("channel_hits", ["vector"])),
            }

        for item in keyword_chunks:
            key = (str(item["doc_id"]), str(item["chunk_id"]))
            current = merged.get(
                key,
                {
                    **item,
                    "vector_score": float(item.get("vector_score", 0.0)),
                    "keyword_score": float(item.get("keyword_score", item.get("score", 0.0))),
                    "channel_hits": [],
                },
            )
            current["keyword_score"] = max(
                float(current.get("keyword_score", 0.0)),
                float(item.get("keyword_score", item.get("score", 0.0))),
            )
            if item.get("keyword_trace"):
                current["keyword_trace"] = item["keyword_trace"]
            if not current.get("doc_name") and item.get("doc_name"):
                current["doc_name"] = item["doc_name"]
            for field in (
                "title_tks",
                "important_kwd",
                "important_tks",
                "question_kwd",
                "question_tks",
                "retrieval_metadata_trace",
            ):
                if not current.get(field) and item.get(field):
                    current[field] = item[field]
            hits = set(current.get("channel_hits", []))
            hits.add("keyword")
            current["channel_hits"] = sorted(hits)
            merged[key] = current

        fused_rows: list[dict] = []
        for item in merged.values():
            vector_score = float(item.get("vector_score", 0.0))
            keyword_score = float(item.get("keyword_score", 0.0))
            # Respect the configured channel weights. The earlier max override
            # let keyword-dominant candidates bypass keyword_weight entirely;
            # a keyword-only candidate received its full keyword score.
            fused_score = vector_weight * vector_score + keyword_weight * keyword_score
            fused_score = min(1.0, max(0.0, fused_score))
            fused_rows.append(
                {
                    **item,
                    "fused_score": fused_score,
                    "score": fused_score,
                }
            )

        fused_rows.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("keyword_score", 0.0)),
                -chunk_order(row),
                float(row.get("vector_score", 0.0)),
            ),
            reverse=True,
        )
        return fused_rows