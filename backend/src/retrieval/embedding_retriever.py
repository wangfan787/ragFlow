from __future__ import annotations

import logging
import math

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from backend.src.config.settings import settings
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.models import build_embeddings

logger = logging.getLogger("mvp_api")


class EmbeddingRetriever:
    def __init__(
        self,
        embedding_model: Embeddings | None = None,
        store: ElasticsearchStore | None = None,
    ) -> None:
        self._embedding = embedding_model
        self._store = store or ElasticsearchStore()

    def retrieve(self, query: str, retrieval_config: dict | None = None) -> list[dict]:
        config = retrieval_config or {}
        top_k = int(config.get("top_k", 10))
        filters = config.get("filters") if isinstance(config.get("filters"), dict) else None

        if len(query.encode("utf-8")) > 3072:
            raise ValueError("检索问题不能超过 3072 UTF-8 字节")
        if self._embedding is None:
            self._embedding = build_embeddings()
        embedding = self._embedding
        query_vector = embedding.embed_query(query)
        dimension = getattr(embedding, "dimensions", None) or 1024
        if len(query_vector) != dimension or not all(math.isfinite(x) for x in query_vector) or not any(query_vector):
            raise ValueError("query embedding has invalid dimension, zero or non-finite values")
        model_filters = {
            "embedding_backend": settings.text("MVP_EMBEDDING_BACKEND", "glm"),
            "embedding_model": embedding.model,
            "embedding_dim": len(query_vector),
            "retrieval_eligible": True,
        }
        rows = self._store.vector_search(
            query_vector,
            top_k=top_k,
            filters={**(filters or {}), **model_filters},
        )

        chunks = [
            {
                **row.metadata, "content": row.page_content,
                "vector_score": row.metadata["score"], "keyword_score": 0.0,
                "fused_score": row.metadata["score"], "channel_hits": ["vector"],
            }
            for row in rows
        ]
        logger.info(
            "retrieval.vector query=%r backend=%s filters=%s hits=%d top=%s",
            query,
            {
                "store": "elasticsearch",
                "embedding": settings.text("MVP_EMBEDDING_BACKEND", "glm"),
                "model": embedding.model,
                "dim": len(query_vector),
            },
            filters,
            len(chunks),
            [
                {
                    "doc_id": row["doc_id"],
                    "chunk_id": row["chunk_id"],
                    "doc_name": row.get("doc_name", ""),
                    "vector_score": round(float(row["vector_score"]), 4),
                    "section_path": row.get("section_path", []),
                }
                for row in chunks[:5]
            ],
        )
        return chunks

    def query_by_ids(self, ids: list[str]) -> list[Document]:
        return self._store.query_by_ids(ids)
