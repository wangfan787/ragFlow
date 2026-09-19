"""调用生产 HybridRouter 录制检索 run（不调用 Chat）。

默认录制 vector / keyword / hybrid / hybrid_rerank 四个 baseline；
run JSONL 直接符合 score_run 冻结 schema，可离线反复评分。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.runner import evaluation_dirs
from evaluation.runner.production_adapter import (
    error_run_row,
    retrieval_run_row,
    run_meta,
)
from evaluation.schemas import read_jsonl, write_jsonl
from evaluation.score_run import score_run

# Rerank 默认用规则后端：cross-encoder 需要显式配置中文/多语模型后再开，
# 不能拿英文默认模型对中文语料下结论（算法层QA Q23）。
DEFAULT_RETRIEVAL_VARIANTS: dict[str, dict] = {
    "vector": {"retrieval_mode": "vector"},
    "keyword": {"retrieval_mode": "keyword"},
    "hybrid": {},  # 生产默认：Hybrid Weighted Sum
    "hybrid_rerank": {"rerank_enabled": True, "rerank_backend": "rule", "rerank_top_n": 10},
}


def load_queries(dataset_dir: Path, split: str) -> list[dict]:
    if split not in {"dev", "test"}:
        raise ValueError("检索 run 只按 dev/test 分片录制")
    return [*read_jsonl(dataset_dir / f"queries.{split}.jsonl")]


def run_retrieval_variant(
    dataset_dir: Path,
    stack,
    split: str,
    variant: str,
    retrieval_overrides: dict,
    *,
    runs_dir: Path,
    score_dir: Path | None = None,
    report_dir: Path | None = None,
    index_name: str | None = None,
) -> dict:
    """录制一个检索变体并完成离线评分；返回 {run, score, metrics...}。"""
    from evaluation.report import render_report

    queries = load_queries(dataset_dir, split)
    es_index = index_name or stack.store.index_name
    meta = run_meta(dataset_dir, es_index, variant, retrieval_overrides, stage="retrieval")
    rows = []
    for query in queries:
        started = time.perf_counter()
        try:
            documents, trace = stack.router.retrieve_detailed(query["question"], retrieval_overrides)
            latency = (time.perf_counter() - started) * 1000.0
            rows.append(retrieval_run_row(query, documents, latency, trace, meta))
        except Exception as exc:  # noqa: BLE001 - 错误逐题记录，不静默丢题
            rows.append(error_run_row(query, str(exc), meta))
    run_path = runs_dir / f"{split}_retrieval_{variant}.jsonl"
    write_jsonl(run_path, rows)

    score = score_run(dataset_dir, run_path, split=split)
    result = {
        "variant": variant,
        "split": split,
        "run_path": run_path,
        "score": score,
        "metrics": score["groups"]["overall"]["metrics"],
        "rows": rows,
    }
    if score_dir is not None:
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f"{split}_retrieval_{variant}.score.json"
        score_path.write_text(
            json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        result["score_path"] = score_path
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{split}_retrieval_{variant}.md"
        report_path.write_text(render_report(score), encoding="utf-8", newline="\n")
        result["report_path"] = report_path
    return result


def run_retrieval(
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
    results = []
    for variant, overrides in (variants or DEFAULT_RETRIEVAL_VARIANTS).items():
        results.append(
            run_retrieval_variant(
                dataset_dir, stack, split, variant, overrides,
                runs_dir=runs_dir, score_dir=dirs["scores"], report_dir=dirs["reports"],
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-name", type=str, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--only", nargs="*", help="只运行指定变体（默认全部）")
    args = parser.parse_args()

    from evaluation.runner.production_adapter import build_stack

    variants = DEFAULT_RETRIEVAL_VARIANTS
    if args.only:
        unknown = set(args.only) - set(variants)
        if unknown:
            raise SystemExit(f"未知变体: {sorted(unknown)}，可选: {sorted(variants)}")
        variants = {name: variants[name] for name in args.only}
    stack = build_stack(args.index_name)
    results = run_retrieval(args.dataset.resolve(), stack, args.split, variants)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['variant']:>14}  recall@10={metrics.get('recall@10', 0):.4f} "
            f"mrr@10={metrics.get('mrr@10', 0):.4f} ndcg@10={metrics.get('ndcg@10', 0):.4f} "
            f"-> {result['run_path']}"
        )


if __name__ == "__main__":
    main()
