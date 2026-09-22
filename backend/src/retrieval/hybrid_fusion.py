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
                "vector_score": float(item["vector_score"]),
                "keyword_score": 0.0,
                "channel_hits": ["vector"],
            }

        for item in keyword_chunks:
            key = (str(item["doc_id"]), str(item["chunk_id"]))
            keyword_score = float(item["keyword_score"])
            if key in merged:
                merged[key]["keyword_score"] = keyword_score
                merged[key]["channel_hits"] = ["keyword", "vector"]
                if "keyword_trace" in item:
                    merged[key]["keyword_trace"] = item["keyword_trace"]
            else:
                merged[key] = {
                    **item,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "channel_hits": ["keyword"],
                }

        fused_rows: list[dict] = []
        for item in merged.values():
            vector_score = float(item["vector_score"])
            keyword_score = float(item["keyword_score"])
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
                float(row["score"]),
                float(row["keyword_score"]),
                -chunk_order(row),
                float(row["vector_score"]),
            ),
            reverse=True,
        )
        return fused_rows
