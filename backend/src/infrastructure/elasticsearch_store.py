from __future__ import annotations

from typing import Any

from backend.src.config.settings import settings
from backend.src.contracts import VectorRecord, VectorSearchResult

_KEYWORD_FIELDS = [
    "important_kwd^30",
    "important_tks^20",
    "question_kwd^20",
    "question_tks^20",
    "title_tks^10",
    "doc_name^10",
    "section_path^8",
    "content^2",
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
            "embedding_backend": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "embedding_dim": {"type": "integer"},
            "retrieval_metadata_trace": {"type": "object", "enabled": False},
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

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        dimensions = {len(record.vector) for record in records}
        if len(dimensions) != 1:
            raise ValueError("a bulk request cannot mix embedding dimensions")
        vector_field = self._ensure_index(dimensions.pop())

        operations = []
        for record in records:
            operations.append({"index": {"_index": self.index_name, "_id": record.id}})
            operations.append(
                {
                    **record.payload,
                    "chunk_id": record.id,
                    "doc_id": record.doc_id,
                    "embedding_dim": len(record.vector),
                    vector_field: record.vector,
                }
            )
        response = self.client.bulk(operations=operations, refresh="wait_for")
        if response.get("errors"):
            raise RuntimeError("Elasticsearch bulk indexing failed")

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        knn: dict[str, Any] = {
            "field": self._vector_field(len(query_vector)),
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(100, top_k * 10),
        }
        if filter_clauses := self._filters(filters):
            knn["filter"] = filter_clauses
        response = self.client.search(
            index=self.index_name,
            knn=knn,
            size=top_k,
            source_excludes=["q_*_vec"],
            ignore_unavailable=True,
            allow_no_indices=True,
        )
        return [
            VectorSearchResult(
                id=str(hit["_source"].get("chunk_id") or hit["_id"]),
                doc_id=str(hit["_source"].get("doc_id", "")),
                score=max(0.0, min(1.0, float(hit.get("_score") or 0.0))),
                payload=dict(hit["_source"]),
            )
            for hit in response.get("hits", {}).get("hits", [])
        ]

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
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
        if filter_clauses := self._filters(filters):
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
        return [
            {
                **hit["_source"],
                "chunk_id": str(hit["_source"].get("chunk_id") or hit["_id"]),
                "doc_id": str(hit["_source"].get("doc_id", "")),
                "keyword_score": (
                    float(hit.get("_score") or 0.0) / max_score if max_score else 0.0
                ),
                "keyword_trace": {
                    "backend": "elasticsearch",
                    "raw_bm25_score": float(hit.get("_score") or 0.0),
                },
            }
            for hit in hits
        ]

    def query_by_ids(self, ids: list[str]) -> list[VectorRecord]:
        if not ids or not self.client.indices.exists(index=self.index_name):
            return []
        response = self.client.mget(index=self.index_name, ids=ids)
        records = []
        for item in response.get("docs", []):
            if not item.get("found"):
                continue
            payload = {
                field: value
                for field, value in item["_source"].items()
                if not (field.startswith("q_") and field.endswith("_vec"))
            }
            records.append(
                VectorRecord(
                    id=str(payload.get("chunk_id") or item["_id"]),
                    doc_id=str(payload.get("doc_id", "")),
                    vector=[],
                    payload=payload,
                )
            )
        return records