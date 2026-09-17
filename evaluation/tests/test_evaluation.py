from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.build_dataset import BuildConfig, TASKS, build_dataset
from evaluation.report import render_report
from evaluation.schemas import read_jsonl, write_jsonl
from evaluation.score_run import score_run
from evaluation.validate_dataset import validate_dataset


def _source_fixture(root: Path) -> Path:
    source = root / "CRUD_RAG"
    split_dir = source / "data" / "crud_split"
    distractor_dir = source / "data" / "80000_docs"
    split_dir.mkdir(parents=True)
    distractor_dir.mkdir(parents=True)
    rows = {}
    for task_index, task in enumerate(TASKS, start=1):
        doc_count = task_index
        task_rows = []
        for row_index in range(3):
            row = {
                "ID": f"event-{task_index}-{row_index}",
                "event": f"事件 {task_index}-{row_index}",
                "questions": f"任务 {task_index} 问题 {row_index}",
                "answers": f"任务 {task_index} 答案 {row_index}",
            }
            for doc_index in range(1, doc_count + 1):
                row[f"news{doc_index}"] = f"正例 {task_index}-{row_index}-{doc_index} 的独立事实"
            task_rows.append(row)
        rows[task] = task_rows
    (split_dir / "split_merged.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    distractors = "\n".join(f"干扰新闻 {index} 包含任务主题但事实不同" for index in range(60)) + "\n"
    (distractor_dir / "part-1").write_text(distractors, encoding="utf-8")
    return source


def _build_fixture(tmp_path: Path) -> Path:
    source = _source_fixture(tmp_path)
    output = tmp_path / "mini"
    build_dataset(
        source,
        output,
        BuildConfig(seed=42, per_task=2, dev_per_task=1, corpus_size=20, hard_negative_count=3),
    )
    return output


def test_builder_is_deterministic_and_dataset_is_valid(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = BuildConfig(seed=42, per_task=2, dev_per_task=1, corpus_size=20, hard_negative_count=3)
    first_manifest = build_dataset(source, first, config)
    second_manifest = build_dataset(source, second, config)

    assert first_manifest == second_manifest
    for name in first_manifest["files"]:
        assert first_manifest["files"][name] == second_manifest["files"][name]
    result = validate_dataset(first)
    assert result["valid"] is True
    assert result["corpus"] == 20
    assert result["queries"] == 6
    assert result["dev"] == 3
    assert result["test"] == 3
    queries = [*read_jsonl(first / "queries.dev.jsonl"), *read_jsonl(first / "queries.test.jsonl")]
    assert len({row["event_id"] for row in queries}) == 6


def test_perfect_run_scores_one_and_renders_report(tmp_path: Path) -> None:
    dataset = _build_fixture(tmp_path)
    queries = [*read_jsonl(dataset / "queries.dev.jsonl"), *read_jsonl(dataset / "queries.test.jsonl")]
    relevant = {}
    for row in read_jsonl(dataset / "qrels.jsonl"):
        relevant.setdefault(row["query_id"], []).append(row["doc_id"])
    corpus_ids = [row["doc_id"] for row in read_jsonl(dataset / "corpus.jsonl")]
    run_rows = []
    for query in queries:
        positives = relevant[query["query_id"]]
        ranking = positives + [doc_id for doc_id in corpus_ids if doc_id not in positives][:10 - len(positives)]
        run_rows.append(
            {
                "query_id": query["query_id"],
                "retrieved": [
                    {"doc_id": doc_id, "rank": rank, "score": 1 / rank, "channel": "fixture"}
                    for rank, doc_id in enumerate(ranking, start=1)
                ],
                "evidence_doc_ids": positives,
                "answer": query["answer"],
                "citations": [{"doc_id": doc_id} for doc_id in positives],
                "status": "ok",
                "latency_ms": {"total": 10.0},
                "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.01},
            }
        )
    run_path = tmp_path / "run.jsonl"
    write_jsonl(run_path, run_rows)
    score = score_run(dataset, run_path)
    metrics = score["groups"]["overall"]["metrics"]
    for name in (
        "hit@1", "hit@3", "hit@5", "hit@10", "recall@3", "recall@5", "recall@10",
        "mrr@10", "ndcg@10", "evidence_recall", "evidence_precision",
        "citation_validity", "citation_document_precision",
    ):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["recall@1"] < 1.0  # 多文档问题在 K=1 时不可能覆盖全部正例。
    assert score["groups"]["overall"]["latency_ms"]["total"]["p95"] == pytest.approx(10.0)
    report = render_report(score)
    assert "# crud_rag_mini_v1 评测报告" in report
    assert "Faithfulness" in report


def test_scorer_rejects_incomplete_run(tmp_path: Path) -> None:
    dataset = _build_fixture(tmp_path)
    query = next(read_jsonl(dataset / "queries.dev.jsonl"))
    run_path = tmp_path / "incomplete.jsonl"
    write_jsonl(
        run_path,
        [
            {
                "query_id": query["query_id"],
                "retrieved": [],
                "evidence_doc_ids": [],
                "answer": "",
                "citations": [],
                "status": "no_evidence",
            }
        ],
    )
    with pytest.raises(ValueError, match="incomplete"):
        score_run(dataset, run_path, split="dev")
