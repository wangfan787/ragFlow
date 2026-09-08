from __future__ import annotations

from typing import Any

from backend.src.config.settings import settings
from langchain_core.documents import Document

_KEYWORD_FIELDS = [
    "doc_name^2",
    "section_path^2",
    "content",
]


class ElasticsearchStore:
    """MVP search store: chunks, BM25 fields, and vectors all live in ES."""

    def __init__(self, client=None, index_name: str | None = None) -> None:
        self._client = client
        self.index_name = index_name or settings.text("MVP_ELASTICSEARCH_INDEX", "rag-mvp-chunks")
        self.url = settings.text("MVP_ELASTICSEARCH_URL", "http://localhost:9200")
        self.timeout = settings.integer("MVP_ELASTICSEARCH_TIMEOUT", 30)

    @property
    def client(self):
        if self._client is None:
            from elasticsearch import Elasticsearch

            self._client = Elasticsearch(self.url, request_timeout=self.timeout)
        return self._client

    def _vector_field(self, dimension: int) -> str:
        if dimension <= 0:
            raise ValueError("embedding vector dimension must be > 0")
        return f"q_{dimension}_vec"

    def _base_mapping(self) -> dict[str, Any]:
        return {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "content": {"type": "text"},
            "section_path": {"type": "text"},
            "page_no": {"type": "integer"},
            "title_tks": {"type": "text"},
            "important_kwd": {"type": "text"},
            "important_tks": {"type": "text"},
            "question_kwd": {"type": "text"},
            "question_tks": {"type": "text"},
            "chunk_role": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "child_ids": {"type": "keyword"},
            "chunk_order": {"type": "integer"},
            "retrieval_eligible": {"type": "boolean"},
            "source_span": {"type": "object", "enabled": False},
            "source_block_ids": {"type": "keyword"},
            "parent_char_start": {"type": "integer"},
            "parent_char_end": {"type": "integer"},
            "embedding_backend": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "embedding_dim": {"type": "integer"},
            "retrieval_metadata_trace": {"type": "object", "enabled": False},
            "embedding_profile": {"type": "keyword"},
            "embedding_input_tokens": {"type": "integer"},
            "embedding_removed_features": {"type": "keyword"},
            "chunk_profile_version": {"type": "keyword"},
            "chunk_profile_hash": {"type": "keyword"},
            "index_schema_version": {"type": "keyword"},
        }

    def _ensure_index(self, dimension: int) -> str:
        vector_field = self._vector_field(dimension)
        vector_mapping = {
            "type": "dense_vector",
            "dims": dimension,
            "index": True,
            "similarity": "cosine",
        }
        if not self.client.indices.exists(index=self.index_name):
            properties = self._base_mapping()
            properties[vector_field] = vector_mapping
            self.client.indices.create(
                index=self.index_name,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
                mappings={"properties": properties},
            )
            return vector_field

        mapping = self.client.indices.get_mapping(index=self.index_name)
        properties = mapping[self.index_name]["mappings"].get("properties", {})
        if vector_field not in properties:
            self.client.indices.put_mapping(
                index=self.index_name,
                properties={vector_field: vector_mapping},
            )
        return vector_field

    def _filters(self, filters: dict[str, Any] | None) -> list[dict]:
        clauses = []
        for field, expected in (filters or {}).items():
            target = "doc_name.keyword" if field == "doc_name" else field
            if isinstance(expected, (list, tuple, set)):
                clauses.append({"terms": {target: list(expected)}})
            else:
                clauses.append({"term": {target: expected}})
        return clauses

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self.client.indices.exists(index=self.index_name):
            return
        self.client.delete_by_query(
            index=self.index_name,
            query={"term": {"doc_id": doc_id}},
            conflicts="proceed",
            refresh=True,
        )

    def delete_stale_by_doc_id(self, doc_id: str, keep_ids: list[str]) -> None:
        if not keep_ids or not self.client.indices.exists(index=self.index_name):
            return
        self.client.delete_by_query(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [{"term": {"doc_id": doc_id}}],
                    "must_not": [{"ids": {"values": keep_ids}}],
                }
            },
            conflicts="proceed",
            refresh=True,
        )

    def upsert(self, records: list[Document], vectors: dict[str, list[float]]) -> None:
        if not records:
            return
        dimensions = {len(vector) for vector in vectors.values()}
        if len(dimensions) != 1:
            raise ValueError("a bulk request must contain exactly one Child embedding dimension")
        vector_field = self._ensure_index(dimensions.pop())

        operations = []
        for record in records:
            chunk_id = record.metadata["chunk_id"]
            operations.append({"index": {"_index": self.index_name, "_id": chunk_id}})
            source = {**record.metadata, "content": record.page_content}
            vector = vectors.get(chunk_id)
            if bool(source.get("retrieval_eligible")) != (vector is not None):
                raise ValueError("only Child documents may have vectors")
            if vector is not None:
                source["embedding_dim"] = len(vector)
                source[vector_field] = vector
            operations.append(source)
        response = self.client.bulk(operations=operations, refresh="wait_for")
        if response.get("errors"):
            failures = [
                item
                for item in response.get("items", [])
                if int(next(iter(item.values())).get("status", 500)) >= 300
            ]
            raise RuntimeError(f"Elasticsearch bulk indexing failed items={failures[:3]}")

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        knn: dict[str, Any] = {
            "field": self._vector_field(len(query_vector)),
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(100, top_k * 10),
        }
        system_filters = {**(filters or {}), "retrieval_eligible": True}
        if filter_clauses := self._filters(system_filters):
            knn["filter"] = filter_clauses
        response = self.client.search(
            index=self.index_name,
            knn=knn,
            size=top_k,
            source_excludes=["q_*_vec"],
            ignore_unavailable=True,
            allow_no_indices=True,
        )
        documents = []
        for hit in response.get("hits", {}).get("hits", []):
            document = self._document(hit)
            document.metadata["score"] = max(0.0, min(1.0, float(hit.get("_score") or 0.0)))
            documents.append(document)
        return documents

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": _KEYWORD_FIELDS,
                        "type": "best_fields",
                    }
                }
            ]
        }
        system_filters = {**(filters or {}), "retrieval_eligible": True}
        if filter_clauses := self._filters(system_filters):
            bool_query["filter"] = filter_clauses
        response = self.client.search(
            index=self.index_name,
            query={"bool": bool_query},
            size=top_k,
            source_excludes=["q_*_vec"],
            ignore_unavailable=True,
            allow_no_indices=True,
        )
        hits = response.get("hits", {}).get("hits", [])
        max_score = max((float(hit.get("_score") or 0.0) for hit in hits), default=0.0)
        documents = []
        for hit in hits:
            document = self._document(hit)
            raw_score = float(hit.get("_score") or 0.0)
            document.metadata.update(
                keyword_score=raw_score / max_score if max_score else 0.0,
                keyword_trace={"backend": "elasticsearch", "raw_bm25_score": raw_score},
            )
            documents.append(document)
        return documents

    @staticmethod
    def _document(hit: dict) -> Document:
        source = hit["_source"]
        metadata = {
            field: value for field, value in source.items()
            if field != "content" and not (field.startswith("q_") and field.endswith("_vec"))
        }
        metadata["chunk_id"] = str(source.get("chunk_id") or hit["_id"])
        metadata["doc_id"] = str(source.get("doc_id", ""))
        return Document(page_content=str(source.get("content", "")), metadata=metadata)

    def query_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids or not self.client.indices.exists(index=self.index_name):
            return []
        response = self.client.mget(index=self.index_name, ids=ids)
        return [self._document(item) for item in response.get("docs", []) if item.get("found")]
