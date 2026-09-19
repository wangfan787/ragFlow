"""调用生产 QAService 录制完整问答 run（真实回答 / 引用 / usage / 阶段耗时）。

默认变体覆盖检索 baseline × 证据模式（child_only / window / full_parent）；
每个变体使用请求级 retrieval_config / qa_config，Trace 原样落盘。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.runner import evaluation_dirs
from evaluation.runner.production_adapter import (
    NO_EVIDENCE_CODES,
    error_run_row,
    qa_run_row,
    run_meta,
)
from evaluation.runner.run_retrieval import load_queries
from evaluation.schemas import write_jsonl
from evaluation.score_run import score_run

# QA run 要支付生成费用：默认只在锁定检索配置后比较证据模式；
# 检索维度的 A/B 全部放在 run_retrieval（不调 Chat）。
DEFAULT_QA_VARIANTS: dict[str, dict] = {
    "hybrid_child_only": {
        "retrieval": {},
        "qa": {"evidence_mode": "child_only"},
    },
    "hybrid_window": {
        "retrieval": {},
        "qa": {"evidence_mode": "window"},
    },
    "hybrid_full_parent": {
        "retrieval": {},
        "qa": {"evidence_mode": "full_parent"},
    },
}


def run_qa_variant(
    dataset_dir: Path,
    stack,
    split: str,
    variant: str,
    variant_spec: dict,
    *,
    runs_dir: Path,
    score_dir: Path | None = None,
    report_dir: Path | None = None,
) -> dict:
    from evaluation.report import render_report

    queries = load_queries(dataset_dir, split)
    retrieval_overrides = dict(variant_spec.get("retrieval") or {})
    qa_overrides = dict(variant_spec.get("qa") or {})
    meta = run_meta(
        dataset_dir, stack.store.index_name, variant,
        retrieval_overrides, qa_overrides, stage="qa",
    )
    rows = []
    for query in queries:
        try:
            payload = stack.qa.query(
                query["question"],
                retrieval_config=retrieval_overrides or None,
                qa_config=qa_overrides or None,
            )
            rows.append(qa_run_row(query, payload, meta))
        except Exception as exc:  # noqa: BLE001 - 逐题录制失败语义
            code = getattr(exc, "code", "") or ""
            status = "no_evidence" if code in NO_EVIDENCE_CODES else "error"
            rows.append(error_run_row(query, f"{code or type(exc).__name__}: {exc}", meta, status))
    run_path = runs_dir / f"{split}_qa_{variant}.jsonl"
    write_jsonl(run_path, rows)

    score = score_run(dataset_dir, run_path, split=split)
    result = {
        "variant": variant, "split": split, "run_path": run_path,
        "score": score, "metrics": score["groups"]["overall"]["metrics"], "rows": rows,
    }
    if score_dir is not None:
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f"{split}_qa_{variant}.score.json"
        score_path.write_text(
            json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        result["score_path"] = score_path
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{split}_qa_{variant}.md"
        report_path.write_text(render_report(score), encoding="utf-8", newline="\n")
        result["report_path"] = report_path
    return result


def run_qa(
    dataset_dir: Path,
    stack,
    split: str,
    variants: dict[str, dict] | None = None,
    *,
    runs_dir: Path | None = None,
) -> list[dict]:
    dirs = evaluation_dirs()
    runs_dir = runs_dir or dirs["runs"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    return [
        run_qa_variant(
            dataset_dir, stack, split, name, spec,
            runs_dir=runs_dir, score_dir=dirs["scores"], report_dir=dirs["reports"],
        )
        for name, spec in (variants or DEFAULT_QA_VARIANTS).items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-name", type=str, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--only", nargs="*", help="只运行指定变体（默认全部）")
    args = parser.parse_args()

    from evaluation.runner.production_adapter import build_stack

    variants = DEFAULT_QA_VARIANTS
    if args.only:
        unknown = set(args.only) - set(variants)
        if unknown:
            raise SystemExit(f"未知变体: {sorted(unknown)}，可选: {sorted(variants)}")
        variants = {name: variants[name] for name in args.only}
    stack = build_stack(args.index_name)
    results = run_qa(args.dataset.resolve(), stack, args.split, variants)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['variant']:>20}  evidence_recall={metrics.get('evidence_recall', 0):.4f} "
            f"citation_validity={metrics.get('citation_validity', 0):.4f} "
            f"-> {result['run_path']}"
        )


if __name__ == "__main__":
    main()
