from __future__ import annotations

import logging

from backend.src.contracts import EmbeddingModel, SearchStore, VectorRecord
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.embedding_factory import build_embedding_model
from backend.src.retrieval.metadata_fields import retrieval_metadata_fields

logger = logging.getLogger("mvp_api")


class EmbeddingRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store: SearchStore | None = None,
    ) -> None:
        self._embedding = embedding_model or build_embedding_model()
        self._store = store or ElasticsearchStore()

    def retrieve(self, query: str, retrieval_config: dict | None = None) -> list[dict]:
        config = retrieval_config or {}
        top_k = int(config.get("top_k", 10))
        filters = config.get("filters") if isinstance(config.get("filters"), dict) else None

        embedding = self._embedding
        query_vector = embedding.encode([query])[0]
        model_filters = {
            "embedding_backend": getattr(embedding, "backend_name", ""),
            "embedding_model": getattr(embedding, "model_name", ""),
            "embedding_dim": len(query_vector),
        }
        rows = self._store.vector_search(
            query_vector,
            top_k=top_k,
            filters={**(filters or {}), **model_filters},
        )

        chunks: list[dict] = []
        for row in rows:
            payload = dict(row.payload)
            chunks.append(
                {
                    "chunk_id": row.id,
                    "doc_id": row.doc_id,
                    "content": str(payload.get("content", "")),
                    "score": row.score,
                    "vector_score": row.score,
                    "keyword_score": 0.0,
                    "fused_score": row.score,
                    "doc_name": str(payload.get("doc_name", "")),
                    **retrieval_metadata_fields(payload),
                    "embedding_backend": str(payload.get("embedding_backend", "")),
                    "embedding_model": str(payload.get("embedding_model", "")),
                    "embedding_dim": payload.get("embedding_dim"),
                    "channel_hits": ["vector"],
                }
            )
        logger.info(
            "retrieval.vector query=%r backend=%s filters=%s hits=%d top=%s",
            query,
            {
                "store": "elasticsearch",
                "embedding": getattr(embedding, "backend_name", ""),
                "model": getattr(embedding, "model_name", ""),
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

    def query_by_ids(self, ids: list[str]) -> list[VectorRecord]:
        return self._store.query_by_ids(ids)