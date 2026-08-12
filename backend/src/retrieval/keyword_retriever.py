from __future__ import annotations

import logging

from backend.src.contracts import SearchStore
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.retrieval.metadata_fields import retrieval_metadata_fields

logger = logging.getLogger("mvp_api")


class KeywordRetriever:
    def __init__(self, store: SearchStore | None = None) -> None:
        self._store = store or ElasticsearchStore()

    def retrieve(self, query: str, retrieval_config: dict | None = None) -> list[dict]:
        config = retrieval_config or {}
        top_k = int(config.get("top_k", 10))
        filters = config.get("filters") if isinstance(config.get("filters"), dict) else None
        hits = self._store.keyword_search(
            query,
            top_k=top_k,
            filters={**(filters or {}), "retrieval_eligible": True},
        )

        rows = []
        for hit in hits:
            score = float(hit.get("keyword_score", 0.0))
            rows.append(
                {
                    "chunk_id": str(hit["chunk_id"]),
                    "doc_id": str(hit["doc_id"]),
                    "content": str(hit.get("content", "")),
                    "score": score,
                    "vector_score": 0.0,
                    "keyword_score": score,
                    "fused_score": score,
                    "doc_name": str(hit.get("doc_name", "")),
                    **retrieval_metadata_fields(hit),
                    "channel_hits": ["keyword"],
                    "keyword_trace": dict(hit.get("keyword_trace", {})),
                }
            )
        logger.info("retrieval.keyword query=%r hits=%d", query, len(rows))
        return rows
