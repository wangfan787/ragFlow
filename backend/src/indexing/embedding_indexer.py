"""只对子块正文嵌入；Document 保存内容/来源，向量单独交给存储。"""
import logging
import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.src.config.settings import settings
from backend.src.indexing.embedding_text_builder import EmbeddingTextBuilder
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.models import build_embeddings

logger = logging.getLogger("mvp_api")


class EmbeddingIndexer:
    def __init__(
        self, embedding_model: Embeddings | None = None,
        store: ElasticsearchStore | None = None, text_builder: EmbeddingTextBuilder | None = None,
    ) -> None:
        self._embedding = embedding_model
        self._store = store or ElasticsearchStore()
        self._text_builder = text_builder or EmbeddingTextBuilder()

    def index(self, doc_id: str, chunks: list[Document], doc_name: str | None = None) -> int:
        if not chunks:
            return 0
        children = [chunk for chunk in chunks if chunk.metadata.get("retrieval_eligible")]
        if not children:
            raise ValueError("document produced no retrieval-eligible Child chunks")
        built = [self._text_builder.build(chunk) for chunk in children]
        if self._embedding is None:
            self._embedding = build_embeddings()
        vectors = self._embedding.embed_documents([item.text for item in built])
        if len(vectors) != len(children):
            raise ValueError("embedding response count does not match Child count")
        dimension = getattr(self._embedding, "dimensions", None) or 1024
        if any(len(v) != dimension or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
            raise ValueError("embedding response has invalid dimension, zero or non-finite values")
        vector_by_id = {chunk.metadata["chunk_id"]: vector for chunk, vector in zip(children, vectors)}
        build_by_id = {chunk.metadata["chunk_id"]: result for chunk, result in zip(children, built)}
        records = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            chunk_id = metadata["chunk_id"]
            vector = vector_by_id.get(chunk_id)
            result = build_by_id.get(chunk_id)
            eligible = bool(metadata.get("retrieval_eligible"))
            metadata.update(
                doc_id=doc_id, doc_name=metadata.get("doc_name") or doc_name or "",
                embedding_backend=settings.text('embedding.backend') if eligible else "",
                embedding_model=self._embedding.model if eligible else "",
                embedding_dim=len(vector) if vector is not None else None,
                embedding_profile=result.profile if result else None,
                embedding_input_tokens=result.token_count if result else None,
                index_schema_version="rag-mvp-v2",
            )
            records.append(Document(page_content=chunk.page_content, metadata=metadata))
        # Publish the replacement before removing stale IDs, as before.
        self._store.upsert(records, vector_by_id)
        self._store.delete_stale_by_doc_id(doc_id, [record.metadata["chunk_id"] for record in records])
        logger.info("embedding.indexed doc_id=%s records=%d children=%d", doc_id, len(records), len(children))
        return len(records)

    def delete_document(self, doc_id: str) -> None:
        """删除该文档在索引中的全部 Parent/Child 记录（文档删除接口调用）。"""
        self._store.delete_by_doc_id(doc_id)
