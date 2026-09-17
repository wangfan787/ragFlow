"""Score a recorded RAG run with deterministic retrieval/evidence metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluation import DATASET_ID
from evaluation.build_dataset import DEFAULT_OUTPUT, TASKS
from evaluation.schemas import finite_number, read_jsonl, sha256_file
from evaluation.validate_dataset import validate_dataset

CUTOFFS = (1, 3, 5, 10)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _query_metrics(row: dict, relevant: set[str]) -> dict[str, float]:
    ranking = [item["doc_id"] for item in row["retrieved"]]
    metrics: dict[str, float] = {}
    for cutoff in CUTOFFS:
        found = relevant & set(ranking[:cutoff])
        metrics[f"hit@{cutoff}"] = float(bool(found))
        metrics[f"recall@{cutoff}"] = len(found) / len(relevant)
    first_rank = next((index for index, doc_id in enumerate(ranking[:10], start=1) if doc_id in relevant), None)
    metrics["mrr@10"] = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(1.0 / math.log2(index + 1) for index, doc_id in enumerate(ranking[:10], start=1) if doc_id in relevant)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, min(10, len(relevant)) + 1))
    metrics["ndcg@10"] = dcg / idcg if idcg else 0.0

    evidence = set(row["evidence_doc_ids"])
    metrics["evidence_recall"] = len(evidence & relevant) / len(relevant)
    metrics["evidence_precision"] = len(evidence & relevant) / len(evidence) if evidence else 0.0
    citations = row["citation_doc_ids"]
    metrics["citation_validity"] = sum(doc_id in evidence for doc_id in citations) / len(citations) if citations else 0.0
    metrics["citation_document_precision"] = sum(doc_id in relevant for doc_id in citations) / len(citations) if citations else 0.0
    return metrics


def _normalize_run_row(row: dict, query_ids: set[str], corpus_ids: set[str], line_number: int) -> dict:
    required = {"query_id", "retrieved", "evidence_doc_ids", "answer", "citations", "status"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"run row {line_number}: missing fields {sorted(missing)}")
    if row["query_id"] not in query_ids:
        raise ValueError(f"run row {line_number}: unknown query_id")
    if row["status"] not in {"ok", "no_evidence", "error"}:
        raise ValueError(f"run row {line_number}: invalid status")
    if not isinstance(row["answer"], str) or not isinstance(row["retrieved"], list):
        raise ValueError(f"run row {line_number}: answer/retrieved type mismatch")

    retrieved_ids = []
    for expected_rank, item in enumerate(row["retrieved"], start=1):
        if not isinstance(item, dict) or {"doc_id", "rank"} - set(item):
            raise ValueError(f"run row {line_number}: invalid retrieved item")
        if item["doc_id"] not in corpus_ids or item["rank"] != expected_rank:
            raise ValueError(f"run row {line_number}: unknown doc or non-contiguous rank")
        if "score" in item:
            finite_number(item["score"], "retrieved.score")
        retrieved_ids.append(item["doc_id"])
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError(f"run row {line_number}: duplicate retrieved doc")

    evidence = row["evidence_doc_ids"]
    if not isinstance(evidence, list) or len(evidence) != len(set(evidence)):
        raise ValueError(f"run row {line_number}: invalid/duplicate evidence_doc_ids")
    if any(doc_id not in corpus_ids for doc_id in evidence):
        raise ValueError(f"run row {line_number}: unknown evidence doc")

    if not isinstance(row["citations"], list):
        raise ValueError(f"run row {line_number}: citations must be a list")
    citation_ids = []
    for citation in row["citations"]:
        doc_id = citation if isinstance(citation, str) else citation.get("doc_id") if isinstance(citation, dict) else None
        if not isinstance(doc_id, str) or doc_id not in corpus_ids:
            raise ValueError(f"run row {line_number}: invalid citation doc")
        citation_ids.append(doc_id)

    latency = row.get("latency_ms", {})
    usage = row.get("usage", {})
    if not isinstance(latency, dict) or not isinstance(usage, dict):
        raise ValueError(f"run row {line_number}: latency_ms/usage must be objects")
    for field, value in latency.items():
        if finite_number(value, f"latency_ms.{field}") < 0:
            raise ValueError(f"run row {line_number}: negative latency")
    for field, value in usage.items():
        if finite_number(value, f"usage.{field}") < 0:
            raise ValueError(f"run row {line_number}: negative usage")
    return {**row, "citation_doc_ids": citation_ids}


def _aggregate(rows: list[dict]) -> dict:
    metric_names = list(rows[0]["metrics"]) if rows else []
    metrics = {name: mean(row["metrics"][name] for row in rows) for name in metric_names}
    latency_fields = sorted({field for row in rows for field in row.get("latency_ms", {})})
    latency = {}
    for field in latency_fields:
        values = [float(row["latency_ms"][field]) for row in rows if field in row.get("latency_ms", {})]
        latency[field] = {"available": len(values), "p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}
    usage_fields = sorted({field for row in rows for field in row.get("usage", {})})
    usage = {
        field: {
            "available": sum(field in row.get("usage", {}) for row in rows),
            "total": sum(float(row.get("usage", {}).get(field, 0)) for row in rows),
        }
        for field in usage_fields
    }
    return {"queries": len(rows), "metrics": metrics, "latency_ms": latency, "usage": usage}


def score_run(dataset_dir: Path, run_path: Path, split: str = "all") -> dict:
    validate_dataset(dataset_dir)
    if split not in {"all", "dev", "test"}:
        raise ValueError("split must be all, dev, or test")
    query_rows = [*read_jsonl(dataset_dir / "queries.dev.jsonl"), *read_jsonl(dataset_dir / "queries.test.jsonl")]
    queries = {row["query_id"]: row for row in query_rows if split == "all" or row["split"] == split}
    corpus_ids = {row["doc_id"] for row in read_jsonl(dataset_dir / "corpus.jsonl")}
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(dataset_dir / "qrels.jsonl"):
        if row["query_id"] in queries:
            qrels[row["query_id"]].add(row["doc_id"])

    runs = {}
    for line_number, row in enumerate(read_jsonl(run_path), start=1):
        normalized = _normalize_run_row(row, set(queries), corpus_ids, line_number)
        query_id = normalized["query_id"]
        if query_id in runs:
            raise ValueError(f"run row {line_number}: duplicate query_id")
        runs[query_id] = normalized
    missing = set(queries) - set(runs)
    if missing:
        raise ValueError(f"run is incomplete for split={split}: missing {len(missing)} queries")

    per_query = []
    for query_id in sorted(queries):
        row = runs[query_id]
        per_query.append(
            {
                "query_id": query_id,
                "task": queries[query_id]["task"],
                "status": row["status"],
                "metrics": _query_metrics(row, qrels[query_id]),
                "latency_ms": row.get("latency_ms", {}),
                "usage": row.get("usage", {}),
            }
        )
    groups = {"overall": _aggregate(per_query)}
    for task in TASKS:
        groups[task] = _aggregate([row for row in per_query if row["task"] == task])
    return {
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "run_file": str(run_path),
        "run_sha256": sha256_file(run_path),
        "split": split,
        "groups": groups,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_run(args.dataset.resolve(), args.run.resolve(), args.split)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
