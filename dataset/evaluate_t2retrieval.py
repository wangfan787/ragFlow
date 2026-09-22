"""Evaluate v2 artifacts through Child candidates -> Family mean -> document."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.src.config.settings import settings

DATASET_DIR = (ROOT / settings.text("dataset.t2_dir")).expanduser().resolve()
ARTIFACT_DIR = DATASET_DIR / "embeddings-v2"
SCHEMA_VERSION = "t2-production-rag-v2"
ADAPTER_ALGORITHM_VERSION = "production-adapter-v2.3-child-body"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_artifact(name: str):
    metadata_path = ARTIFACT_DIR / f"{name}.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    progress = json.loads(
        (ARTIFACT_DIR / f"{name}.embedding.progress.json").read_text(encoding="utf-8")
    )
    prepared = json.loads(
        (ARTIFACT_DIR / f"{name}.prepare.progress.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema_version") != SCHEMA_VERSION or not metadata.get("complete"):
        raise ValueError(f"{name} final metadata is incomplete or unsupported")
    if (
        not progress.get("complete")
        or int(progress.get("next_vector_row", -1)) != int(metadata["vector_rows"])
        or not prepared.get("complete")
    ):
        raise ValueError(f"{name} preparation/embedding has not completed")
    preparation_contract = {
        "schema_version": metadata.get("schema_version"),
        "algorithm_version": metadata.get("algorithm_version"),
        "input_fingerprint": metadata.get("input_fingerprint"),
        "profile_hash": metadata.get("profile_hash"),
        "source_file": metadata.get("source_file"),
        "source_file_sha256": metadata.get("source_file_sha256"),
        "source_total_rows": metadata.get("source_total_rows"),
        "source_start_row": metadata.get("source_start_row"),
        "source_end_row": metadata.get("source_end_row"),
        "source_selection": metadata.get("source_selection"),
        "artifact_rows": metadata.get("vector_rows"),
        "mapping_bytes": metadata.get("mapping_bytes"),
        "mapping_sha256": metadata.get("mapping_sha256"),
        "excluded_documents": metadata.get("excluded_documents"),
        "exclusions_file": metadata.get("exclusions_file"),
        "exclusions_bytes": metadata.get("exclusions_bytes"),
        "exclusions_sha256": metadata.get("exclusions_sha256"),
    }
    if any(prepared.get(key) != value for key, value in preparation_contract.items()):
        raise ValueError(f"{name} preparation/mapping contract does not match final vectors")
    for key in (
        "algorithm_version", "input_fingerprint", "profile_hash", "source_file",
        "source_file_sha256", "source_total_rows", "source_start_row", "source_end_row",
        "source_selection", "model", "backend", "dimension", "shape",
        "excluded_documents", "exclusions_file", "exclusions_bytes", "exclusions_sha256",
        "vectors_bytes", "vectors_sha256",
    ):
        if progress.get(key) != metadata.get(key):
            raise ValueError(f"{name} progress/metadata contract mismatch: {key}")
    if metadata.get("algorithm_version") != ADAPTER_ALGORITHM_VERSION:
        raise ValueError(f"{name} artifact algorithm is not {ADAPTER_ALGORITHM_VERSION}")
    mmap_path = ARTIFACT_DIR / f"{name}.f32.mmap"
    expected_bytes = int(np.prod(metadata["shape"])) * np.dtype(metadata["dtype"]).itemsize
    if mmap_path.stat().st_size != expected_bytes:
        raise ValueError(f"{name} mmap byte size does not match declared shape/dtype")
    if (
        int(metadata.get("vectors_bytes", -1)) != expected_bytes
        or _sha256_file(mmap_path) != metadata.get("vectors_sha256")
    ):
        raise ValueError(f"{name} vector digest/byte contract mismatch")
    vectors = np.memmap(
        mmap_path,
        dtype=metadata["dtype"],
        mode="r",
        shape=tuple(metadata["shape"]),
    )
    mapping_path = ARTIFACT_DIR / metadata["mapping_file"]
    if (
        mapping_path.stat().st_size != int(metadata["mapping_bytes"])
        or _sha256_file(mapping_path) != metadata["mapping_sha256"]
    ):
        raise ValueError(f"{name} mapping digest/byte contract mismatch")
    mapping = []
    with mapping_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping.append(
                {
                    "chunk_id": row["chunk_id"],
                    "parent_id": row.get("parent_id"),
                    "doc_id": row["doc_id"],
                    "chunk_order": row.get("chunk_order", 0),
                    "embedding_text": row.get("embedding_text", ""),
                }
            )
    if len(mapping) != vectors.shape[0]:
        raise ValueError(f"{name} vector/mapping row mismatch")
    exclusions_path = ARTIFACT_DIR / metadata["exclusions_file"]
    if (
        exclusions_path.stat().st_size != int(metadata["exclusions_bytes"])
        or _sha256_file(exclusions_path) != metadata["exclusions_sha256"]
    ):
        raise ValueError(f"{name} exclusions digest/byte contract mismatch")
    excluded_rows: set[int] = set()
    excluded_doc_ids: set[str] = set()
    with exclusions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            exclusion = json.loads(line)
            required = {"document_row", "doc_id", "reason", "source_length", "source_sha256"}
            if not isinstance(exclusion, dict) or not required.issubset(exclusion):
                raise ValueError(f"{name} exclusion line {line_number} is invalid")
            document_row = int(exclusion["document_row"])
            doc_id = str(exclusion["doc_id"])
            if not 0 <= document_row < int(metadata["document_rows"]):
                raise ValueError(f"{name} exclusion document_row is out of range")
            if document_row in excluded_rows or doc_id in excluded_doc_ids:
                raise ValueError(f"{name} exclusions contain duplicate documents")
            if int(exclusion["source_length"]) < 0:
                raise ValueError(f"{name} exclusion source_length cannot be negative")
            source_sha256 = str(exclusion["source_sha256"])
            if len(source_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in source_sha256
            ):
                raise ValueError(f"{name} exclusion source_sha256 is invalid")
            excluded_rows.add(document_row)
            excluded_doc_ids.add(doc_id)
    if len(excluded_rows) != int(metadata["excluded_documents"]):
        raise ValueError(f"{name} exclusion row count mismatch")
    if name == "queries" and excluded_rows:
        raise ValueError("query artifacts cannot contain excluded records")
    return metadata, vectors, mapping


def _validate_evaluation_scope(
    corpus_meta: dict,
    query_meta: dict,
    query_map: list[dict],
    expected_query_count: int,
) -> None:
    corpus_path = DATASET_DIR / "corpus.parquet"
    queries_path = DATASET_DIR / "queries.parquet"
    corpus_rows = pq.read_metadata(corpus_path).num_rows
    query_rows = pq.read_metadata(queries_path).num_rows
    if (
        corpus_meta.get("source_selection") != "all"
        or int(corpus_meta.get("source_start_row", -1)) != 0
        or int(corpus_meta.get("source_end_row", -1)) != corpus_rows
        or int(corpus_meta.get("source_total_rows", -1)) != corpus_rows
        or int(corpus_meta.get("document_rows", -1)) != corpus_rows
    ):
        raise ValueError("evaluation requires an all-rows corpus artifact")
    if (
        # 10k 子集场景：queries 文件本身就是被抽中的范围，全量嵌入时记为 "all"；
        # 全量语料做 prefix 实验时仍为 "prefix"。两者都必须通过数量、指纹与 ID 校验。
        query_meta.get("source_selection") not in ("prefix", "all")
        or int(query_meta.get("source_start_row", -1)) != 0
        or int(query_meta.get("source_end_row", -1)) != expected_query_count
        or int(query_meta.get("source_total_rows", -1)) != query_rows
        or int(query_meta.get("document_rows", -1)) != expected_query_count
    ):
        raise ValueError(f"evaluation requires the first {expected_query_count} query rows")
    if _sha256_file(corpus_path) != corpus_meta.get("source_file_sha256"):
        raise ValueError("corpus source file digest does not match artifact")
    if _sha256_file(queries_path) != query_meta.get("source_file_sha256"):
        raise ValueError("queries source file digest does not match artifact")
    expected_query_ids = [
        str(value)
        for value in pq.read_table(queries_path, columns=["_id"]).column("_id")[:expected_query_count].to_pylist()
    ]
    actual_query_ids = [str(row["doc_id"]) for row in query_map]
    if actual_query_ids != expected_query_ids:
        raise ValueError("query artifact IDs are not the requested deterministic prefix")


def _qrels() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in pq.read_table(DATASET_DIR / "qrels.parquet").to_pylist():
        qid = str(row["query-id"] if "query-id" in row else row.get("query_id"))
        doc_id = str(row["corpus-id"] if "corpus-id" in row else row.get("corpus_id"))
        result.setdefault(qid, {})[doc_id] = float(row.get("score", 1.0))
    return result


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _family_document_ranking(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    corpus_map: list[dict],
    *,
    aggregation: str,
    top_n: int,
    top_k: int,
) -> list[str]:
    families: dict[str, list[float]] = {}
    family_doc: dict[str, str] = {}
    for index, score in zip(candidate_indices, candidate_scores):
        if int(index) < 0:
            continue
        row = corpus_map[int(index)]
        family_id = str(row.get("parent_id") or row["chunk_id"])
        families.setdefault(family_id, []).append(float(score))
        family_doc[family_id] = str(row["doc_id"])
    document_families: dict[str, list[float]] = {}
    for family_id, scores in families.items():
        document_families.setdefault(family_doc[family_id], []).append(sum(scores) / len(scores))
    document_scores = {}
    for doc_id, scores in document_families.items():
        ranked = sorted(scores, reverse=True)
        document_scores[doc_id] = (
            ranked[0] if aggregation == "max" else sum(ranked[:top_n]) / len(ranked[:top_n])
        )
    return [
        doc_id
        for doc_id, _ in sorted(document_scores.items(), key=lambda item: item[1], reverse=True)[
            :top_k
        ]
    ]


def evaluate(
    *,
    top_k: int,
    candidate_top_k: int,
    query_batch: int,
    corpus_batch: int,
    aggregation: str,
    top_n: int,
    engine: str,
    expected_query_count: int | None = 1000,
    enforce_scope: bool = True,
    recall_k: list[int] | None = None,
    rerank_model: str | None = None,
    rerank_batch_size: int = 64,
) -> dict:
    for name, value in {
        "top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "query_batch": query_batch,
        "corpus_batch": corpus_batch,
        "top_n": top_n,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    cutoffs = sorted({int(k) for k in (recall_k or (1, 3, 5, 10))} | {top_k})
    if any(k <= 0 for k in cutoffs):
        raise ValueError("recall cutoffs must be positive")
    ranking_depth = max(top_k, 10, *cutoffs)
    if candidate_top_k < ranking_depth:
        raise ValueError("candidate_top_k must be >= max(top_k, 10, *recall_k)")
    corpus_meta, corpus_vectors, corpus_map = _load_artifact("corpus")
    query_meta, query_vectors, query_map = _load_artifact("queries")
    for key in ("model", "backend", "dimension"):
        if corpus_meta[key] != query_meta[key]:
            raise ValueError(f"corpus/query embedding contracts differ: {key}")
    if enforce_scope:
        if expected_query_count is None or expected_query_count <= 0:
            raise ValueError("expected_query_count must be positive when scope enforcement is enabled")
        _validate_evaluation_scope(corpus_meta, query_meta, query_map, expected_query_count)
    qrels = _qrels()
    reranker = None
    if rerank_model:
        # 复用生产 CrossEncoderReranker；模型下载走 HF_ENDPOINT 镜像。
        from backend.src.infrastructure.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=rerank_model, batch_size=rerank_batch_size)
    rankings: dict[str, list[str]] = {}
    candidate_count = min(candidate_top_k, corpus_vectors.shape[0])

    ann_index = None
    if engine == "faiss-hnsw":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "full T2 evaluation requires faiss-cpu in the project agent environment; "
                "use --engine exact only for small correctness fixtures"
            ) from exc
        dimension = int(corpus_vectors.shape[1])
        ann_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        ann_index.hnsw.efConstruction = 80
        ann_index.hnsw.efSearch = max(128, candidate_count * 2)
        for c_start in range(0, corpus_vectors.shape[0], corpus_batch):
            c_end = min(c_start + corpus_batch, corpus_vectors.shape[0])
            ann_index.add(np.ascontiguousarray(_normalize(np.asarray(corpus_vectors[c_start:c_end]))))

    for q_start in range(0, query_vectors.shape[0], query_batch):
        q_end = min(q_start + query_batch, query_vectors.shape[0])
        queries = np.ascontiguousarray(_normalize(np.asarray(query_vectors[q_start:q_end])))
        if ann_index is not None:
            best_scores, best_indices = ann_index.search(queries, candidate_count)
        else:
            best_scores = np.full((len(queries), candidate_count), -np.inf, dtype=np.float32)
            best_indices = np.full((len(queries), candidate_count), -1, dtype=np.int64)
            for c_start in range(0, corpus_vectors.shape[0], corpus_batch):
                c_end = min(c_start + corpus_batch, corpus_vectors.shape[0])
                scores = queries @ _normalize(np.asarray(corpus_vectors[c_start:c_end])).T
                indices = np.broadcast_to(
                    np.arange(c_start, c_end, dtype=np.int64), scores.shape
                )
                combined_scores = np.concatenate((best_scores, scores), axis=1)
                combined_indices = np.concatenate((best_indices, indices), axis=1)
                selected = np.argpartition(combined_scores, -candidate_count, axis=1)[
                    :, -candidate_count:
                ]
                best_scores = np.take_along_axis(combined_scores, selected, axis=1)
                best_indices = np.take_along_axis(combined_indices, selected, axis=1)
        for local_q in range(len(queries)):
            qid = str(query_map[q_start + local_q]["doc_id"])
            candidates = [
                {
                    "index": int(index),
                    "vector_score": float(score),
                    "chunk_order": corpus_map[int(index)].get("chunk_order", 0),
                    "content": corpus_map[int(index)].get("embedding_text", ""),
                }
                for index, score in zip(best_indices[local_q], best_scores[local_q])
                if int(index) >= 0
            ]
            if reranker is not None:
                candidates = reranker.rerank(
                    str(query_map[q_start + local_q].get("embedding_text", "")), candidates
                )
                rerank_indices = np.array([row["index"] for row in candidates], dtype=np.int64)
                rerank_scores = np.array([row["score"] for row in candidates], dtype=np.float64)
                rankings[qid] = _family_document_ranking(
                    rerank_indices,
                    rerank_scores,
                    corpus_map,
                    aggregation=aggregation,
                    top_n=top_n,
                    top_k=ranking_depth,
                )
                continue
            order = np.argsort(best_scores[local_q])[::-1]
            rankings[qid] = _family_document_ranking(
                best_indices[local_q, order],
                best_scores[local_q, order],
                corpus_map,
                aggregation=aggregation,
                top_n=top_n,
                top_k=ranking_depth,
            )

    recall_sums = {k: 0.0 for k in cutoffs}
    mrr_sum = ndcg_sum = 0.0
    evaluated = 0
    relevant_pairs = 0
    retrievable_relevant_pairs = 0
    retrievable_doc_ids = {str(row["doc_id"]) for row in corpus_map}
    for qid, relevant in qrels.items():
        if qid not in rankings:
            continue
        evaluated += 1
        relevant_pairs += len(relevant)
        retrievable_relevant_pairs += sum(doc_id in retrievable_doc_ids for doc_id in relevant)
        ranked = rankings[qid]
        for cutoff in cutoffs:
            cutoff_hits = [doc_id for doc_id in ranked[:cutoff] if doc_id in relevant]
            recall_sums[cutoff] += len(cutoff_hits) / max(1, len(relevant))
        top_ten_hits = [index for index, doc_id in enumerate(ranked[:10]) if doc_id in relevant]
        mrr_sum += 1.0 / (top_ten_hits[0] + 1) if top_ten_hits else 0.0
        dcg = sum(
            (2 ** relevant[doc_id] - 1) / math.log2(index + 2)
            for index, doc_id in enumerate(ranked[:10])
            if doc_id in relevant
        )
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum((2**score - 1) / math.log2(index + 2) for index, score in enumerate(ideal))
        ndcg_sum += dcg / idcg if idcg else 0.0
    if not evaluated:
        raise ValueError("no qrels matched query artifacts")
    return {
        "artifact_schema": corpus_meta["schema_version"],
        "candidate_top_k": candidate_top_k,
        "candidate_engine": engine,
        "rerank_model": rerank_model,
        "family_aggregation": "mean(hit_children)",
        "document_aggregation": aggregation,
        "top_n": top_n if aggregation == "topn-mean" else None,
        "evaluated_queries": evaluated,
        "query_scope": f"{query_meta.get('source_selection')}:{int(query_meta['document_rows'])}",
        "query_artifact_rows": int(query_meta["document_rows"]),
        "corpus_document_rows": int(corpus_meta["document_rows"]),
        "excluded_corpus_documents": int(corpus_meta["excluded_documents"]),
        "qrels_relevance_pairs": relevant_pairs,
        "retrievable_qrels_pairs": retrievable_relevant_pairs,
        "qrels_coverage": retrievable_relevant_pairs / relevant_pairs if relevant_pairs else 0.0,
        **{f"Recall@{k}": recall_sums[k] / evaluated for k in cutoffs},
        "MRR@10": mrr_sum / evaluated,
        "nDCG@10": ndcg_sum / evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--corpus-batch", type=int, default=4096)
    parser.add_argument("--aggregation", choices=("max", "topn-mean"), default="max")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--engine", choices=("faiss-hnsw", "exact"), default="faiss-hnsw")
    parser.add_argument("--expected-query-count", type=int, default=1000)
    parser.add_argument("--rerank-model", default=None, help="cross-encoder 模型名；不传则关闭 rerank")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    args = parser.parse_args()
    for name in ("top_k", "candidate_top_k", "query_batch", "corpus_batch", "top_n"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
