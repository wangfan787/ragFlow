"""Validate a frozen CRUD-RAG Mini dataset and its reproducibility contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import DATASET_ID, SCHEMA_VERSION
from evaluation.build_dataset import DEFAULT_OUTPUT, TASKS, TASK_DOCS
from evaluation.schemas import normalize_text, read_jsonl, sha256_file, stable_id

REQUIRED_FILES = {
    "corpus.jsonl",
    "queries.dev.jsonl",
    "queries.test.jsonl",
    "qrels.jsonl",
    "references.jsonl",
}


def _require_fields(row: dict, fields: set[str], label: str) -> None:
    missing = fields - set(row)
    if missing:
        raise ValueError(f"{label}: missing fields {sorted(missing)}")


def validate_dataset(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dataset_id/schema_version")
    if set(manifest.get("files", {})) != REQUIRED_FILES:
        raise ValueError("manifest must describe exactly the required dataset artifacts")
    for name, expected in manifest["files"].items():
        path = dataset_dir / name
        if not path.is_file():
            raise ValueError(f"missing artifact: {name}")
        actual = {
            "bytes": path.stat().st_size,
            "lines": sum(1 for line in path.open("rb") if line.strip()),
            "sha256": sha256_file(path),
        }
        if actual != expected:
            raise ValueError(f"artifact contract mismatch for {name}: {actual} != {expected}")

    corpus_rows = list(read_jsonl(dataset_dir / "corpus.jsonl"))
    corpus: dict[str, dict] = {}
    normalized_docs: set[str] = set()
    for index, row in enumerate(corpus_rows, start=1):
        _require_fields(row, {"doc_id", "text", "source"}, f"corpus row {index}")
        text = normalize_text(row["text"])
        expected_id = stable_id("crud_", text)
        if row["doc_id"] != expected_id:
            raise ValueError(f"corpus row {index}: unstable doc_id")
        if row["doc_id"] in corpus or text in normalized_docs:
            raise ValueError(f"corpus row {index}: duplicate document")
        if row["source"] not in {"positive", "hard_negative", "random_negative"}:
            raise ValueError(f"corpus row {index}: invalid source")
        corpus[row["doc_id"]] = row
        normalized_docs.add(text)

    dev_rows = list(read_jsonl(dataset_dir / "queries.dev.jsonl"))
    test_rows = list(read_jsonl(dataset_dir / "queries.test.jsonl"))
    if any(row.get("split") != "dev" for row in dev_rows):
        raise ValueError("queries.dev.jsonl contains a non-dev row")
    if any(row.get("split") != "test" for row in test_rows):
        raise ValueError("queries.test.jsonl contains a non-test row")
    query_rows = [*dev_rows, *test_rows]
    queries: dict[str, dict] = {}
    event_ids: set[str] = set()
    task_counts = Counter()
    split_counts = Counter()
    task_split_counts = Counter()
    for index, row in enumerate(query_rows, start=1):
        _require_fields(
            row,
            {"query_id", "task", "event_id", "event", "question", "answer", "split"},
            f"query row {index}",
        )
        if row["task"] not in TASKS or row["split"] not in {"dev", "test"}:
            raise ValueError(f"query row {index}: invalid task/split")
        expected_id = stable_id("crud_q_", row["task"], row["event_id"], row["question"])
        if row["query_id"] != expected_id:
            raise ValueError(f"query row {index}: unstable query_id")
        if row["query_id"] in queries or row["event_id"] in event_ids:
            raise ValueError(f"query row {index}: duplicate query/event")
        if not normalize_text(row["question"]) or not normalize_text(row["answer"]):
            raise ValueError(f"query row {index}: empty question/answer")
        queries[row["query_id"]] = row
        event_ids.add(row["event_id"])
        task_counts[row["task"]] += 1
        split_counts[row["split"]] += 1
        task_split_counts[(row["task"], row["split"])] += 1

    qrels: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(read_jsonl(dataset_dir / "qrels.jsonl"), start=1):
        _require_fields(row, {"query_id", "doc_id", "relevance"}, f"qrel row {index}")
        if row["query_id"] not in queries or row["doc_id"] not in corpus:
            raise ValueError(f"qrel row {index}: unknown query/doc")
        if row["relevance"] != 1 or row["doc_id"] in qrels[row["query_id"]]:
            raise ValueError(f"qrel row {index}: invalid relevance or duplicate")
        qrels[row["query_id"]].add(row["doc_id"])
        if corpus[row["doc_id"]]["source"] != "positive":
            raise ValueError(f"qrel row {index}: relevant doc is not labelled positive")

    references = {}
    for index, row in enumerate(read_jsonl(dataset_dir / "references.jsonl"), start=1):
        _require_fields(row, {"query_id", "answer", "expected_doc_ids"}, f"reference row {index}")
        query_id = row["query_id"]
        if query_id not in queries or query_id in references:
            raise ValueError(f"reference row {index}: unknown or duplicate query")
        if normalize_text(row["answer"]) != normalize_text(queries[query_id]["answer"]):
            raise ValueError(f"reference row {index}: answer mismatch")
        if set(row["expected_doc_ids"]) != qrels[query_id]:
            raise ValueError(f"reference row {index}: expected_doc_ids mismatch")
        references[query_id] = row

    if set(qrels) != set(queries) or set(references) != set(queries):
        raise ValueError("every query must have qrels and one reference row")
    for query_id, relevant in qrels.items():
        expected = TASK_DOCS[queries[query_id]["task"]]
        if not 1 <= len(relevant) <= expected:
            raise ValueError(f"{query_id}: relevant doc count is inconsistent with task")

    counts = manifest["counts"]
    actual_counts = {
        "corpus": len(corpus),
        "queries": len(queries),
        "dev": split_counts["dev"],
        "test": split_counts["test"],
        "qrels": sum(len(value) for value in qrels.values()),
        "positive_documents": sum(row["source"] == "positive" for row in corpus.values()),
        "hard_negatives": sum(row["source"] == "hard_negative" for row in corpus.values()),
        "random_negatives": sum(row["source"] == "random_negative" for row in corpus.values()),
        "tasks": {task: task_counts[task] for task in TASKS},
    }
    if counts != actual_counts:
        raise ValueError(f"manifest counts mismatch: {actual_counts} != {counts}")
    per_task = int(manifest["parameters"]["per_task"])
    dev_per_task = int(manifest["parameters"]["dev_per_task"])
    for task in TASKS:
        if task_split_counts[(task, "dev")] != dev_per_task:
            raise ValueError(f"{task}: incorrect dev count")
        if task_split_counts[(task, "test")] != per_task - dev_per_task:
            raise ValueError(f"{task}: incorrect test count")
    if len(corpus) != int(manifest["parameters"]["corpus_size"]):
        raise ValueError("corpus size does not match parameters")
    return {"valid": True, **actual_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset.resolve()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
