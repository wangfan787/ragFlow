from __future__ import annotations

import logging

from backend.src.contracts import EmbeddingModel, SearchStore, VectorRecord
from backend.src.indexing.embedding_text_builder import EmbeddingTextBuilder
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.embedding_factory import build_embedding_model
from backend.src.retrieval.metadata_fields import retrieval_metadata_fields

logger = logging.getLogger("mvp_api")


class EmbeddingIndexer:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store: SearchStore | None = None,
        text_builder: EmbeddingTextBuilder | None = None,
    ) -> None:
        self._embedding = embedding_model or build_embedding_model()
        self._store = store or ElasticsearchStore()
        self._text_builder = text_builder or EmbeddingTextBuilder()

    def index(self, doc_id: str, chunks: list[dict], doc_name: str | None = None) -> int:
        if not chunks:
            return 0
        children = [chunk for chunk in chunks if bool(chunk.get("retrieval_eligible"))]
        if not children:
            raise ValueError("document produced no retrieval-eligible Child chunks")
        model_limit = getattr(self._embedding, "max_input_tokens", None)
        bounded_children = []
        for chunk in children:
            configured_limit = int(
                chunk.get("embedding_input_budget")
                or self._text_builder.profile.input_budget_tokens
            )
            effective_limit = min(configured_limit, int(model_limit)) if model_limit else configured_limit
            bounded_children.append({**chunk, "embedding_input_budget": effective_limit})
        built = [
            self._text_builder.build(chunk, doc_name=doc_name)
            for chunk in bounded_children
        ]
        truncated_before = int(getattr(self._embedding, "truncated_input_count", 0))
        vectors = self._embedding.encode([item.text for item in built])
        truncated_after = int(getattr(self._embedding, "truncated_input_count", 0))
        if truncated_after != truncated_before:
            raise RuntimeError(
                "embedding adapter truncated production input; refusing to publish lossy vectors"
            )
        if len(vectors) != len(children):
            raise ValueError("embedding response count does not match Child count")
        if any(not vector for vector in vectors):
            raise ValueError("embedding response contained an empty vector")
        vector_by_id = {
            str(chunk["chunk_id"]): vector for chunk, vector in zip(children, vectors)
        }
        build_by_id = {str(chunk["chunk_id"]): result for chunk, result in zip(children, built)}

        records: list[VectorRecord] = []
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            vector = vector_by_id.get(chunk_id)
            build_result = build_by_id.get(chunk_id)
            eligible = bool(chunk.get("retrieval_eligible"))
            if eligible != (vector is not None):
                raise RuntimeError("record role and vector presence disagree")
            records.append(
                VectorRecord(
                    id=chunk_id,
                    doc_id=doc_id,
                    vector=vector,
                    payload={
                        **retrieval_metadata_fields(chunk),
                        "chunk_id": chunk_id,
                        "content": str(chunk.get("content", "")),
                        "doc_id": doc_id,
                        "doc_name": str(chunk.get("doc_name") or doc_name or ""),
                        "embedding_backend": (
                            getattr(self._embedding, "backend_name", "") if eligible else ""
                        ),
                        "embedding_model": (
                            getattr(self._embedding, "model_name", "") if eligible else ""
                        ),
                        "embedding_dim": len(vector) if vector is not None else None,
                        "embedding_profile": build_result.profile if build_result else None,
                        "embedding_input_tokens": build_result.token_count if build_result else None,
                        "embedding_removed_features": (
                            list(build_result.removed_features) if build_result else []
                        ),
                        "index_schema_version": "rag-mvp-v2",
                    },
                )
            )

        # Publish the complete replacement before removing stale IDs. A failed
        # bulk therefore leaves the previous searchable generation available.
        self._store.upsert(records)
        self._store.delete_stale_by_doc_id(doc_id, [record.id for record in records])
        logger.info(
            "embedding.indexed doc_id=%s records=%d children=%d parents=%d backend=%s model=%s",
            doc_id,
            len(records),
            len(children),
            len(records) - len(children),
            getattr(self._embedding, "backend_name", ""),
            getattr(self._embedding, "model_name", ""),
        )
        return len(records)
