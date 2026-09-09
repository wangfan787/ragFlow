"""Build resumable T2Retrieval v2 artifacts through production RAG components."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter
from backend.src.chunking.token_counter import count_tokens
from backend.src.config.settings import settings
from backend.src.infrastructure.models import build_embeddings

LOGGER = logging.getLogger("t2retrieval_embedding_v2")
# 默认全量目录；跑 1 万条子集时显式导出 T2_DATASET_DIR，防止误把全量送去 embedding。
DATASET_DIR = Path(os.environ.get("T2_DATASET_DIR") or ROOT / "dataset" / "T2Retrieval")
OUTPUT_DIR = DATASET_DIR / "embeddings-v2"
SCHEMA_VERSION = "t2-production-rag-v2"
ADAPTER_ALGORITHM_VERSION = "production-adapter-v2.3-child-body"


def _read_table(name: str):
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return pq.read_table(path)


def _fingerprint(name: str, total: int, profile_hash: str, source_sha256: str) -> str:
    path = DATASET_DIR / f"{name}.parquet"
    raw = (
        f"{SCHEMA_VERSION}|{ADAPTER_ALGORITHM_VERSION}|{name}|{total}|"
        f"{path.stat().st_size}|{source_sha256}|{profile_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _profile_hash(adapter: ProductionRagAdapter) -> str:
    raw = json.dumps(
        {
            "chunk": asdict(adapter.chunk_config),
            "embedding": adapter.text_builder.profile,
            "algorithm_version": ADAPTER_ALGORITHM_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _durable_flush(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _require_committed_prefix(path: Path, committed_bytes: int, label: str) -> None:
    if committed_bytes < 0:
        raise ValueError(f"{label} committed byte offset cannot be negative")
    if not path.exists():
        if committed_bytes:
            raise ValueError(f"{label} is missing with {committed_bytes} committed bytes")
        return
    actual = path.stat().st_size
    if actual < committed_bytes:
        raise ValueError(
            f"{label} is shorter than committed progress: {actual} < {committed_bytes}"
        )


def _require_exact_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} is missing")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size does not match committed contract")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} digest does not match committed contract")


def _validate_jsonl_count(path: Path, expected: int, label: str) -> None:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} contains invalid JSON at line {rows}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{label} line {rows} must be a JSON object")
    if rows != expected:
        raise ValueError(f"{label} row count mismatch: {rows} != {expected}")


def _validate_exclusions(path: Path, expected: int, total_documents: int) -> None:
    rows: set[int] = set()
    doc_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"exclusions contains invalid JSON at line {line_number}") from exc
            required = {"document_row", "doc_id", "reason", "source_length", "source_sha256"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError(f"exclusions line {line_number} is missing required fields")
            document_row = int(payload["document_row"])
            doc_id = str(payload["doc_id"])
            if not 0 <= document_row < total_documents:
                raise ValueError(f"exclusions document_row out of range: {document_row}")
            if document_row in rows or doc_id in doc_ids:
                raise ValueError(f"duplicate exclusion document at line {line_number}")
            if int(payload["source_length"]) < 0:
                raise ValueError(f"negative exclusion source_length at line {line_number}")
            digest = str(payload["source_sha256"])
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid exclusion source_sha256 at line {line_number}")
            rows.add(document_row)
            doc_ids.add(doc_id)
    if len(rows) != expected:
        raise ValueError(f"exclusion row count mismatch: {len(rows)} != {expected}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_manifest(name: str, adapter: ProductionRagAdapter, limit: int | None) -> tuple[Path, dict]:
    table = _read_table(name)
    source_path = DATASET_DIR / f"{name}.parquet"
    source_sha256 = _sha256_file(source_path)
    total_docs = table.num_rows if limit is None else min(limit, table.num_rows)
    profile_hash = _profile_hash(adapter)
    fingerprint = _fingerprint(name, total_docs, profile_hash, source_sha256)
    manifest_path = OUTPUT_DIR / f"{name}.chunks.jsonl"
    exclusions_path = OUTPUT_DIR / f"{name}.exclusions.jsonl"
    progress_path = OUTPUT_DIR / f"{name}.prepare.progress.json"
    progress = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ADAPTER_ALGORITHM_VERSION,
        "input_fingerprint": fingerprint,
        "profile_hash": profile_hash,
        "source_file": source_path.name,
        "source_file_sha256": source_sha256,
        "source_total_rows": table.num_rows,
        "source_start_row": 0,
        "source_end_row": total_docs,
        "source_selection": "all" if total_docs == table.num_rows else "prefix",
        "next_document_row": 0,
        "artifact_rows": 0,
        "manifest_bytes": 0,
        "excluded_documents": 0,
        "exclusions_bytes": 0,
        "total_documents": total_docs,
    }
    if progress_path.exists():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            existing.get("input_fingerprint") != fingerprint
            or existing.get("algorithm_version") != ADAPTER_ALGORITHM_VERSION
        ):
            raise ValueError(f"existing {name} preparation is incompatible with current input/profile")
        progress.update(existing)

    next_document_row = int(progress["next_document_row"])
    artifact_rows = int(progress["artifact_rows"])
    excluded_documents = int(progress["excluded_documents"])
    if not 0 <= next_document_row <= total_docs:
        raise ValueError(f"invalid {name} next_document_row: {next_document_row}")
    if artifact_rows < 0 or not 0 <= excluded_documents <= next_document_row:
        raise ValueError(f"invalid {name} preparation counters")

    _require_committed_prefix(manifest_path, int(progress["manifest_bytes"]), f"{name} manifest")
    _require_committed_prefix(
        exclusions_path,
        int(progress["exclusions_bytes"]),
        f"{name} exclusions",
    )
    if progress.get("complete"):
        if next_document_row != total_docs:
            raise ValueError(f"complete {name} preparation has incomplete document progress")
        _require_exact_file(
            manifest_path,
            int(progress["mapping_bytes"]),
            str(progress["mapping_sha256"]),
            f"{name} manifest",
        )
        _require_exact_file(
            exclusions_path,
            int(progress["exclusions_bytes"]),
            str(progress["exclusions_sha256"]),
            f"{name} exclusions",
        )
        _validate_jsonl_count(manifest_path, artifact_rows, f"{name} manifest")
        _validate_exclusions(exclusions_path, excluded_documents, total_docs)
        return manifest_path, progress

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if manifest_path.exists() else "w+b"
    exclusions_mode = "r+b" if exclusions_path.exists() else "w+b"
    with manifest_path.open(mode) as handle, exclusions_path.open(exclusions_mode) as exclusions:
        handle.truncate(int(progress["manifest_bytes"]))
        handle.seek(0, 2)
        exclusions.truncate(int(progress["exclusions_bytes"]))
        exclusions.seek(0, 2)
        _durable_flush(handle)
        _durable_flush(exclusions)
        if next_document_row:
            _validate_jsonl_count(manifest_path, artifact_rows, f"{name} manifest")
            _validate_exclusions(exclusions_path, excluded_documents, total_docs)
        for offset in range(int(progress["next_document_row"]), total_docs):
            row = table.slice(offset, 1).to_pylist()[0]
            item_id = str(row.get("_id"))
            records: list[dict]
            exclusion: dict | None = None
            if name == "corpus":
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                try:
                    records = adapter.child_inputs(doc_id=item_id, doc_name=title, text=text)
                except ValueError as exc:
                    if str(exc) != "no parseable content found":
                        raise
                    records = []
                    exclusion = {
                        "document_row": offset,
                        "doc_id": item_id,
                        "reason": "no_parseable_content",
                        "source_length": len(text),
                        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
            else:
                query = str(row.get("text") or "").strip()
                records = [
                    {
                        "chunk_id": f"query::{item_id}",
                        "doc_id": item_id,
                        "parent_id": None,
                        "chunk_order": 0,
                        "source_span": None,
                        "parent_char_start": 0,
                        "parent_char_end": len(query),
                        "chunk_profile_version": "query-v1",
                        "chunk_profile_hash": profile_hash,
                        "embedding_profile": "query-v1",
                        "embedding_input_tokens": count_tokens(query),
                        "embedding_text": query,
                    }
                ]
            if not records:
                if name != "corpus":
                    raise ValueError(f"query produced no embedding record: {item_id}")
                if exclusion is None:
                    exclusion = {
                        "document_row": offset,
                        "doc_id": item_id,
                        "reason": "no_retrieval_eligible_child",
                        "source_length": len(text),
                        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                exclusions.write((json.dumps(exclusion, ensure_ascii=False) + "\n").encode("utf-8"))
                LOGGER.warning(
                    "excluded %s document_row=%d doc_id=%s reason=%s",
                    name,
                    offset,
                    item_id,
                    exclusion["reason"],
                )
            for record in records:
                handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            _durable_flush(handle)
            _durable_flush(exclusions)
            progress.update(
                {
                    "next_document_row": offset + 1,
                    "artifact_rows": int(progress["artifact_rows"]) + len(records),
                    "manifest_bytes": handle.tell(),
                    "excluded_documents": int(progress["excluded_documents"])
                    + (1 if exclusion is not None else 0),
                    "exclusions_bytes": exclusions.tell(),
                }
            )
            _atomic_json(progress_path, progress)
    progress["complete"] = True
    _validate_jsonl_count(manifest_path, int(progress["artifact_rows"]), f"{name} manifest")
    _validate_exclusions(
        exclusions_path,
        int(progress["excluded_documents"]),
        total_docs,
    )
    progress["mapping_bytes"] = manifest_path.stat().st_size
    progress["mapping_sha256"] = _sha256_file(manifest_path)
    progress["exclusions_file"] = exclusions_path.name
    progress["exclusions_sha256"] = _sha256_file(exclusions_path)
    _atomic_json(progress_path, progress)
    return manifest_path, progress


def _manifest_batches(path: Path, start: int, batch_size: int):
    batch_texts: list[str] = []
    batch_rows: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if row_number < start:
                continue
            record = json.loads(line)
            batch_texts.append(str(record["embedding_text"]))
            batch_rows.append(row_number)
            if len(batch_texts) >= batch_size:
                yield batch_rows, batch_texts
                batch_rows, batch_texts = [], []
        if batch_texts:
            yield batch_rows, batch_texts


def _embed_manifest(name: str, model, manifest_path: Path, prepared: dict, batch_size: int) -> None:
    total = int(prepared["artifact_rows"])
    dimension = int(getattr(model, "dimensions", 0) or 0)
    if dimension <= 0:
        raise ValueError("MVP_EMBEDDING_DIMENSIONS must be set for persistent artifacts")
    prefix = OUTPUT_DIR / name
    mmap_path = prefix.with_suffix(".f32.mmap")
    metadata_path = prefix.with_suffix(".metadata.json")
    pending_metadata_path = prefix.with_suffix(".metadata.pending.json")
    progress_path = prefix.with_suffix(".embedding.progress.json")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ADAPTER_ALGORITHM_VERSION,
        "input_fingerprint": prepared["input_fingerprint"],
        "profile_hash": prepared["profile_hash"],
        "split": name,
        "source_file": prepared["source_file"],
        "source_file_sha256": prepared["source_file_sha256"],
        "source_total_rows": int(prepared["source_total_rows"]),
        "source_start_row": int(prepared["source_start_row"]),
        "source_end_row": int(prepared["source_end_row"]),
        "source_selection": str(prepared["source_selection"]),
        "document_rows": prepared["total_documents"],
        "vector_rows": total,
        "shape": [total, dimension],
        "dtype": "float32",
        "dimension": dimension,
        "model": model.model,
        "backend": settings.text("MVP_EMBEDDING_BACKEND", "glm"),
        "mapping_file": manifest_path.name,
        "mapping_bytes": int(prepared["mapping_bytes"]),
        "mapping_sha256": str(prepared["mapping_sha256"]),
        "excluded_documents": int(prepared["excluded_documents"]),
        "exclusions_file": str(prepared["exclusions_file"]),
        "exclusions_bytes": int(prepared["exclusions_bytes"]),
        "exclusions_sha256": str(prepared["exclusions_sha256"]),
        "score_mapping": "chunk vectors map to corpus doc_id through mapping_file",
    }
    progress = {**contract, "next_vector_row": 0}
    if progress_path.exists():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        comparable = (
            "schema_version", "algorithm_version", "input_fingerprint", "profile_hash",
            "source_file", "source_file_sha256", "source_total_rows", "source_start_row",
            "source_end_row", "source_selection", "shape", "model", "backend",
            "vector_rows", "mapping_bytes", "mapping_sha256", "excluded_documents",
            "exclusions_file", "exclusions_bytes", "exclusions_sha256",
        )
        if any(existing.get(key) != contract.get(key) for key in comparable):
            raise ValueError(f"existing {name} vector progress violates artifact contract")
        progress.update(existing)

    start = int(progress["next_vector_row"])
    if not 0 <= start <= total:
        raise ValueError(f"invalid {name} next_vector_row: {start}")
    expected_vector_bytes = total * dimension * np.dtype("float32").itemsize
    _require_exact_file(
        manifest_path,
        int(contract["mapping_bytes"]),
        str(contract["mapping_sha256"]),
        f"{name} manifest",
    )
    _require_exact_file(
        OUTPUT_DIR / str(contract["exclusions_file"]),
        int(contract["exclusions_bytes"]),
        str(contract["exclusions_sha256"]),
        f"{name} exclusions",
    )
    if start:
        if not mmap_path.exists():
            raise ValueError(f"{name} mmap is missing with {start} committed vector rows")
        if mmap_path.stat().st_size != expected_vector_bytes:
            raise ValueError(f"{name} mmap byte size does not match committed vector shape")
    if progress.get("complete"):
        if start != total:
            raise ValueError(f"complete {name} embedding has incomplete vector progress")
        _require_exact_file(
            mmap_path,
            int(progress["vectors_bytes"]),
            str(progress["vectors_sha256"]),
            f"{name} mmap",
        )
        final_metadata = {
            **contract,
            "complete": True,
            "vectors_bytes": int(progress["vectors_bytes"]),
            "vectors_sha256": str(progress["vectors_sha256"]),
        }
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(existing_metadata.get(key) != value for key, value in final_metadata.items()):
                raise ValueError(f"existing {name} final metadata violates artifact contract")
        else:
            _atomic_json(metadata_path, final_metadata)
        pending_metadata_path.unlink(missing_ok=True)
        return
    mode = "r+" if start else "w+"
    vectors = np.memmap(mmap_path, dtype="float32", mode=mode, shape=(total, dimension))
    if not progress.get("complete"):
        metadata_path.unlink(missing_ok=True)
    _atomic_json(pending_metadata_path, {**contract, "complete": False})
    for row_numbers, texts in _manifest_batches(manifest_path, start, batch_size):
        matrix = np.asarray(model.embed_documents(texts), dtype="float32")
        expected = (len(texts), dimension)
        if matrix.shape != expected:
            raise RuntimeError(f"embedding shape mismatch: {matrix.shape} != {expected}")
        row_start, row_end = row_numbers[0], row_numbers[-1] + 1
        vectors[row_start:row_end] = matrix
        vectors.flush()
        _fsync_path(mmap_path)
        progress["next_vector_row"] = row_end
        _atomic_json(progress_path, progress)
        LOGGER.info("embedded %s %d/%d", name, row_end, total)
    progress["complete"] = True
    vectors.flush()
    _fsync_path(mmap_path)
    vectors_sha256 = _sha256_file(mmap_path)
    progress["vectors_bytes"] = mmap_path.stat().st_size
    progress["vectors_sha256"] = vectors_sha256
    _atomic_json(progress_path, progress)
    _atomic_json(
        metadata_path,
        {
            **contract,
            "complete": True,
            "vectors_bytes": progress["vectors_bytes"],
            "vectors_sha256": vectors_sha256,
        },
    )
    pending_metadata_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("corpus", "queries", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adapter = ProductionRagAdapter()
    model = None if args.prepare_only else build_embeddings()
    batch_size = args.batch_size or settings.integer("MVP_EMBEDDING_BATCH_SIZE", 16, positive=True)
    splits = ("corpus", "queries") if args.split == "both" else (args.split,)
    for split in splits:
        manifest, prepared = _prepare_manifest(split, adapter, args.limit)
        if model is not None:
            _embed_manifest(split, model, manifest, prepared, batch_size)


if __name__ == "__main__":
    main()
