from __future__ import annotations

import logging

from backend.src.contracts import EmbeddingModel, SearchStore, VectorRecord
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.embedding_factory import build_embedding_model
from backend.src.retrieval.metadata_fields import as_text_list, retrieval_metadata_fields

logger = logging.getLogger("mvp_api")


class EmbeddingIndexer:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store: SearchStore | None = None,
    ) -> None:
        self._embedding = embedding_model or build_embedding_model()
        self._store = store or ElasticsearchStore()

    def index(
        self,
        doc_id: str,
        chunks: list[dict],
        doc_name: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        vectors = self._embedding.encode(
            [self._retrieval_text(chunk, doc_name=doc_name) for chunk in chunks]
        )
        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_doc_name = str(chunk.get("doc_name") or doc_name or "")
            embedding_model = getattr(self._embedding, "model_name", "")
            records.append(
                VectorRecord(
                    id=str(chunk["chunk_id"]),
                    doc_id=doc_id,
                    vector=vector,
                    payload={
                        **retrieval_metadata_fields(chunk),
                        "chunk_id": str(chunk["chunk_id"]),
                        "content": str(chunk.get("content", "")),
                        "doc_id": doc_id,
                        "doc_name": chunk_doc_name,
                        "embedding_backend": getattr(self._embedding, "backend_name", ""),
                        "embedding_model": embedding_model,
                        "embedding_dim": len(vector),
                        # 父子分块字段进 payload：父存完整 content、子存 parent_id，
                        # 为后续"子召回父返回"的检索联动预埋。
                        "chunk_role": str(chunk.get("chunk_role", "parent")),
                        "parent_id": chunk.get("parent_id"),
                        "child_ids": list(chunk.get("child_ids", []) or []),
                        "chunk_order": int(chunk.get("chunk_order", 0)) or None,
                    },
                )
            )

        # A document ingest is a replacement operation. Remove stale chunks
        # from an older parse before publishing the new complete chunk set.
        self._store.delete_by_doc_id(doc_id)
        self._store.upsert(records)
        logger.info(
            "embedding.indexed doc_id=%s chunks=%d backend=%s model=%s dim=%s store=%s",
            doc_id,
            len(records),
            getattr(self._embedding, "backend_name", ""),
            getattr(self._embedding, "model_name", ""),
            len(vectors[0]) if vectors else None,
            "elasticsearch",
        )
        return len(records)

    def _retrieval_text(self, chunk: dict, doc_name: str | None = None) -> str:
        chunk_doc_name = str(chunk.get("doc_name") or doc_name or "")
        section_path = " ".join(as_text_list(chunk.get("section_path", [])))
        questions = "\n".join(as_text_list(chunk.get("question_kwd", [])))
        content = questions or str(chunk.get("content", ""))
        return "\n".join(part for part in [chunk_doc_name, section_path, content] if part)