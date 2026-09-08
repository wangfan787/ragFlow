from __future__ import annotations

import logging

from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore

logger = logging.getLogger("mvp_api")


class KeywordRetriever:
    def __init__(self, store: ElasticsearchStore | None = None) -> None:
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

        rows = [
            {
                **hit.metadata, "content": hit.page_content,
                "score": hit.metadata["keyword_score"], "vector_score": 0.0,
                "fused_score": hit.metadata["keyword_score"], "channel_hits": ["keyword"],
            }
            for hit in hits
        ]
        logger.info("retrieval.keyword query=%r hits=%d", query, len(rows))
        return rows
