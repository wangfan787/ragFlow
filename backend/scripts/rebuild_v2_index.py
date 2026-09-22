"""Rebuild all registered documents into a separate v2 physical ES index.

This command never changes config/*.yaml, swaps an alias, or deletes the
old index. It only builds and validates the explicitly named target index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.apps.services.state_store import list_documents
from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.config.settings import settings
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.models import build_embeddings


def rebuild(index_name: str, *, limit: int | None = None, dry_run: bool = False) -> dict:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    current = settings.text('elasticsearch.index')
    if index_name == current:
        raise ValueError("v2 rebuild target must differ from the active physical index")
    registered = list_documents()
    if not registered:
        raise ValueError("v2 rebuild requires at least one registered document")
    missing = [
        {"doc_id": doc.get("doc_id"), "file_path": doc.get("file_path")}
        for doc in registered
        if not Path(doc.get("file_path", "")).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "v2 rebuild preflight failed; registered source files are missing: "
            + json.dumps(missing, ensure_ascii=False)
        )
    documents = list(registered)
    if limit is not None:
        documents = documents[:limit]
    manifest = {
        "schema_version": "rag-mvp-v2",
        "target_index": index_name,
        "active_index_unchanged": current,
        "switchable": limit is None,
        "registered_document_count": len(registered),
        "documents": [],
    }
    if dry_run:
        manifest["documents"] = [doc["doc_id"] for doc in documents]
        return manifest

    store = ElasticsearchStore(index_name=index_name)
    client = store.client
    if client.indices.exists(index=index_name):
        raise ValueError(
            "v2 rebuild target already exists; choose a new empty physical index name"
        )
    embedding_model = build_embeddings()
    pipeline = IngestionPipeline(
        embedding_indexer=EmbeddingIndexer(
            embedding_model=embedding_model,
            store=store,
        )
    )
    for doc in documents:
        result = pipeline.run(
            str(doc["doc_id"]),
            {
                "file_type": doc["file_type"],
                "file_path": doc["file_path"],
                "doc_name": doc["name"],
            },
            ChunkConfig(),
        )
        manifest["documents"].append(result)

    total = client.count(index=index_name).get("count", 0)
    parents = client.count(
        index=index_name,
        query={"term": {"retrieval_eligible": False}},
    ).get("count", 0)
    children = client.count(
        index=index_name,
        query={"term": {"retrieval_eligible": True}},
    ).get("count", 0)
    expected_total = sum(int(item["indexed_count"]) for item in manifest["documents"])
    if total != expected_total or total != parents + children or (
        documents and (parents == 0 or children == 0)
    ):
        raise RuntimeError("v2 validation failed: Parent/Child record counts are inconsistent")
    per_document = {}
    for doc, result in zip(documents, manifest["documents"]):
        doc_id = str(doc["doc_id"])
        actual = int(
            client.count(index=index_name, query={"term": {"doc_id": doc_id}}).get("count", 0)
        )
        if actual != int(result["indexed_count"]):
            raise RuntimeError(f"v2 validation failed for doc_id={doc_id}: {actual} records")
        per_document[doc_id] = actual

    dimension = int(getattr(embedding_model, "dimensions", 0) or 0)
    if dimension <= 0:
        raise RuntimeError("embedding model dimension is required for v2 validation")
    vector_field = store._vector_field(dimension)
    sample_response = client.search(
        index=index_name,
        query={"term": {"retrieval_eligible": True}},
        size=1,
        source_includes=[
            "chunk_id", "doc_id", "chunk_role", "retrieval_eligible",
            "index_schema_version", "chunk_profile_version", "chunk_profile_hash",
            "embedding_model", "embedding_dim", "source_span", "source_block_ids",
            "parent_id", "parent_char_start", "parent_char_end", vector_field,
        ],
    )
    samples = sample_response.get("hits", {}).get("hits", [])
    if documents and not samples:
        raise RuntimeError("v2 validation failed: no retrievable Child sample")
    if samples:
        sample = samples[0]["_source"]
        required = (
            "chunk_id", "doc_id", "parent_id", "source_span", "source_block_ids",
            "parent_char_start", "parent_char_end", "chunk_profile_hash", vector_field,
        )
        if any(sample.get(field) in (None, "", []) for field in required):
            raise RuntimeError("v2 validation failed: Child sample lacks vector/provenance fields")
        if (
            sample.get("index_schema_version") != "rag-mvp-v2"
            or sample.get("chunk_profile_version") != "parent-child-v2"
            or sample.get("chunk_role") != "child"
            or sample.get("retrieval_eligible") is not True
            or int(sample.get("embedding_dim", 0)) != dimension
            or sample.get("embedding_model") != embedding_model.model
        ):
            raise RuntimeError("v2 validation failed: Child sample schema/model contract mismatch")
        smoke = store.vector_search(
            list(sample[vector_field]),
            top_k=1,
            filters={"doc_id": sample["doc_id"]},
        )
        if not smoke or not smoke[0].metadata.get("retrieval_eligible"):
            raise RuntimeError("v2 validation failed: filtered Child retrieval smoke test failed")
    manifest["validation"] = {
        "total_records": total,
        "parent_context_records": parents,
        "child_vector_records": children,
        "expected_document_ids": sorted(per_document),
        "per_document_records": per_document,
        "embedding_model": embedding_model.model,
        "embedding_dimension": dimension,
        "filtered_retrieval_smoke": bool(samples),
        "eligible_for_config_switch": limit is None,
        "switch_performed": False,
        "old_index_deleted": False,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-name", default="rag-mvp-chunks-v2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    result = rebuild(args.index_name, limit=args.limit, dry_run=args.dry_run)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.manifest:
        args.manifest.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
