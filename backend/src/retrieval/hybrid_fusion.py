from __future__ import annotations

from backend.src.retrieval.ranking import chunk_order

RRF_RANK_CONSTANT = 60


class HybridFusion:
    """等权 RRF；输入为两路按相关性降序排列的 Child 候选。"""

    def fuse(
        self,
        vector_chunks: list[dict],
        keyword_chunks: list[dict],
    ) -> list[dict]:
        merged: dict[tuple[str, str], dict] = {}

        for rank, item in enumerate(vector_chunks, start=1):
            key = (str(item["doc_id"]), str(item["chunk_id"]))
            merged[key] = {
                **item,
                "vector_score": float(item["vector_score"]),
                "keyword_score": 0.0,
                "rrf_score": 1.0 / (RRF_RANK_CONSTANT + rank),
                "channel_hits": ["vector"],
            }

        for rank, item in enumerate(keyword_chunks, start=1):
            key = (str(item["doc_id"]), str(item["chunk_id"]))
            keyword_score = float(item["keyword_score"])
            rank_score = 1.0 / (RRF_RANK_CONSTANT + rank)
            if key in merged:
                merged[key]["keyword_score"] = keyword_score
                merged[key]["rrf_score"] += rank_score
                merged[key]["channel_hits"] = ["keyword", "vector"]
                if "keyword_trace" in item:
                    merged[key]["keyword_trace"] = item["keyword_trace"]
            else:
                merged[key] = {
                    **item,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "rrf_score": rank_score,
                    "channel_hits": ["keyword"],
                }

        fused_rows: list[dict] = []
        for item in merged.values():
            # 除以双路均排第一的理论上限，固定缩放到 0~1，不改变 RRF 排序。
            # 即使一路无命中也使用相同分母；此分数不是相关概率，阈值需重新评测。
            fused_score = item["rrf_score"] / (2.0 / (RRF_RANK_CONSTANT + 1))
            fused_rows.append(
                {
                    **item,
                    "fused_score": fused_score,
                    "score": fused_score,
                }
            )

        fused_rows.sort(
            key=lambda row: (
                -float(row["score"]),
                chunk_order(row),
                str(row["doc_id"]),
                str(row["chunk_id"]),
            ),
        )
        return fused_rows
