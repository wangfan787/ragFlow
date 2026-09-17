"""Build the deterministic CRUD-RAG Mini benchmark without upstream code."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from evaluation import DATASET_ID, SCHEMA_VERSION
from evaluation.schemas import normalize_text, sha256_file, stable_id, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "CRUD_RAG"
DEFAULT_OUTPUT = ROOT / "evaluation" / "datasets" / DATASET_ID
TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")
TASK_DOCS = {"questanswer_1doc": 1, "questanswer_2docs": 2, "questanswer_3docs": 3}
ALGORITHM_VERSION = "crud-mini-builder-v1"
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]")


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 42
    per_task: int = 50
    dev_per_task: int = 10
    corpus_size: int = 1000
    hard_negative_count: int = 150


def _source_commit(source_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _tokens(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    ascii_tokens = {token for token in _ASCII_WORD.findall(normalized) if len(token) > 1}
    chars = _CJK.findall(normalized)
    return ascii_tokens | {"".join(chars[index:index + 2]) for index in range(len(chars) - 1)}


def _select_queries(rows_by_task: dict[str, list[dict]], config: BuildConfig) -> list[dict]:
    rng = random.Random(config.seed)
    used_events: set[str] = set()
    selected: list[dict] = []
    for task in TASKS:
        candidates = list(rows_by_task.get(task, []))
        rng.shuffle(candidates)
        accepted = []
        for row in candidates:
            event_id = str(row.get("ID", "")).strip()
            if not event_id or event_id in used_events:
                continue
            question = normalize_text(row.get("questions", ""))
            answer = normalize_text(row.get("answers", ""))
            expected = [normalize_text(row.get(f"news{i}", "")) for i in range(1, TASK_DOCS[task] + 1)]
            if not question or not answer or any(not text for text in expected):
                continue
            accepted.append((row, event_id, question, answer, expected))
            used_events.add(event_id)
            if len(accepted) == config.per_task:
                break
        if len(accepted) != config.per_task:
            raise ValueError(f"{task}: need {config.per_task} distinct usable events, found {len(accepted)}")
        for index, (row, event_id, question, answer, expected) in enumerate(accepted):
            split = "dev" if index < config.dev_per_task else "test"
            query_id = stable_id("crud_q_", task, event_id, question)
            selected.append(
                {
                    "query_id": query_id,
                    "task": task,
                    "event_id": event_id,
                    "event": normalize_text(row.get("event", "")),
                    "question": question,
                    "answer": answer,
                    "split": split,
                    "reference_texts": expected,
                }
            )
    return selected


def _positive_documents(selected: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    documents: dict[str, dict] = {}
    query_docs: dict[str, list[str]] = {}
    for query in selected:
        doc_ids = []
        for text in query.pop("reference_texts"):
            doc_id = stable_id("crud_", text)
            documents.setdefault(
                doc_id,
                {"doc_id": doc_id, "text": text, "source": "positive"},
            )
            doc_ids.append(doc_id)
        query_docs[query["query_id"]] = list(dict.fromkeys(doc_ids))
    return documents, query_docs


def _iter_distractors(source_dir: Path):
    data_dir = source_dir / "data" / "80000_docs"
    paths = sorted(path for path in data_dir.iterdir() if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no distractor files found under {data_dir}")
    for path in paths:
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line in handle:
                text = normalize_text(line)
                if text:
                    yield text


def _choose_distractors(
    source_dir: Path,
    selected: list[dict],
    positives: dict[str, dict],
    config: BuildConfig,
) -> list[dict]:
    needed = config.corpus_size - len(positives)
    if needed < 0:
        raise ValueError(f"positive documents ({len(positives)}) exceed corpus_size")
    positive_ids = set(positives)
    query_tokens = set().union(*(_tokens(query["question"]) for query in selected))
    candidates: dict[str, tuple[str, float]] = {}
    for text in _iter_distractors(source_dir):
        doc_id = stable_id("crud_", text)
        if doc_id in positive_ids or doc_id in candidates:
            continue
        tokens = _tokens(text)
        overlap = len(tokens & query_tokens)
        score = overlap / math.sqrt(max(1, len(tokens)))
        candidates[doc_id] = (text, score)
    if len(candidates) < needed:
        raise ValueError(f"distractor pool has {len(candidates)} unique rows, need {needed}")

    hard_count = min(config.hard_negative_count, needed)
    hard_ids = [
        doc_id
        for doc_id, _ in sorted(candidates.items(), key=lambda item: (-item[1][1], item[0]))[:hard_count]
    ]
    remaining_ids = sorted(set(candidates) - set(hard_ids))
    rng = random.Random(config.seed + 1)
    random_ids = rng.sample(remaining_ids, needed - hard_count)
    return [
        {"doc_id": doc_id, "text": candidates[doc_id][0], "source": "hard_negative"}
        for doc_id in hard_ids
    ] + [
        {"doc_id": doc_id, "text": candidates[doc_id][0], "source": "random_negative"}
        for doc_id in random_ids
    ]


def _file_record(path: Path) -> dict:
    with path.open("rb") as handle:
        lines = sum(1 for _ in handle)
    return {"bytes": path.stat().st_size, "lines": lines, "sha256": sha256_file(path)}


def build_dataset(source_dir: Path, output_dir: Path, config: BuildConfig = BuildConfig()) -> dict:
    split_path = source_dir / "data" / "crud_split" / "split_merged.json"
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    selected = _select_queries(raw, config)
    positives, query_docs = _positive_documents(selected)
    distractors = _choose_distractors(source_dir, selected, positives, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = sorted([*positives.values(), *distractors], key=lambda row: row["doc_id"])
    queries = [
        {key: query[key] for key in ("query_id", "task", "event_id", "event", "question", "answer", "split")}
        for query in selected
    ]
    qrels = [
        {"query_id": query["query_id"], "doc_id": doc_id, "relevance": 1}
        for query in queries
        for doc_id in query_docs[query["query_id"]]
    ]
    references = [
        {
            "query_id": query["query_id"],
            "answer": query["answer"],
            "expected_doc_ids": query_docs[query["query_id"]],
        }
        for query in queries
    ]
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "queries.dev.jsonl", (row for row in queries if row["split"] == "dev"))
    write_jsonl(output_dir / "queries.test.jsonl", (row for row in queries if row["split"] == "test"))
    write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "references.jsonl", references)
    readme = (
        f"# {DATASET_ID}\n\n"
        "该目录由 `python -m evaluation.build_dataset` 确定性生成，不提交 Git。\n"
        "正例来自 CRUD-RAG 问答行的 news 字段；80k 文档只用于干扰项。\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    data_files = ("corpus.jsonl", "queries.dev.jsonl", "queries.test.jsonl", "qrels.jsonl", "references.jsonl")
    manifest = {
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": {
            "repository": "https://github.com/IAAR-Shanghai/CRUD_RAG.git",
            "commit": _source_commit(source_dir),
            "split_file": str(split_path.relative_to(source_dir)),
            "split_file_sha256": sha256_file(split_path),
        },
        "parameters": {
            "seed": config.seed,
            "per_task": config.per_task,
            "dev_per_task": config.dev_per_task,
            "corpus_size": config.corpus_size,
            "hard_negative_count": config.hard_negative_count,
        },
        "counts": {
            "corpus": len(corpus),
            "queries": len(queries),
            "dev": sum(row["split"] == "dev" for row in queries),
            "test": sum(row["split"] == "test" for row in queries),
            "qrels": len(qrels),
            "positive_documents": len(positives),
            "hard_negatives": sum(row["source"] == "hard_negative" for row in corpus),
            "random_negatives": sum(row["source"] == "random_negative" for row in corpus),
            "tasks": {task: sum(row["task"] == task for row in queries) for task in TASKS},
        },
        "files": {name: _file_record(output_dir / name) for name in data_files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_dataset(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
