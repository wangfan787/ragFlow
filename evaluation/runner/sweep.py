"""Dev-only 单变量参数扫描与配置锁定。

规则：
1. 只在 Dev 分片扫描；每个轴只改一个变量，禁止无边界笛卡尔积；
2. 检索层扫描不调用 Chat；证据层扫描（evidence mode / window tokens）
   才产生生成费用，且只在 Dev；
3. 依据 recall@10 > mrr@10 > ndcg@10 的确定序锁定配置，写入
   evaluation/locks/；锁定后 Test 只运行一次，不再回扫；
4. Rerank 无稳定收益（recall@10 增益 < 1pp 或 mrr@10 下降）则保持
   默认关闭；Weighted Sum 未暴露刻度问题则不实现 RRF（不在本工具做）。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from evaluation.runner import evaluation_dirs, git_commit
from evaluation.runner.production_adapter import (
    error_run_row,
    qa_run_row,
    retrieval_run_row,
    run_meta,
)
from evaluation.runner.run_retrieval import (
    DEFAULT_RETRIEVAL_VARIANTS,
    load_queries,
)
from evaluation.schemas import sha256_file, write_jsonl
from evaluation.score_run import score_run

# Rerank 生效判定：recall@10 绝对增益阈值 + mrr@10 不得下降
RERANK_MIN_RECALL_GAIN = 0.01

# 检索层扫描轴：每条记录只覆盖一个变量（其余沿用生产默认）
SWEEP_AXES: dict[str, list[dict]] = {
    "candidate_top_k": [{"candidate_top_k": value} for value in (10, 30, 50, 100)],
    "top_k": [{"top_k": value} for value in (3, 5, 10)],
    "similarity_threshold": [{"similarity_threshold": value} for value in (0.0, 0.05, 0.1, 0.2)],
    "vector_weight": [{"vector_weight": value} for value in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)],
    "rerank": [
        {"rerank_enabled": True, "rerank_backend": "rule", "rerank_top_n": value}
        for value in (10, 20, 30)
    ],
}

# 证据层扫描（调用 Chat，仅 Dev）：模式 + 窗口 token
EVIDENCE_SWEEP: list[dict] = [
    {"evidence_mode": "child_only"},
    {"evidence_mode": "window", "evidence_window_tokens": 256},
    {"evidence_mode": "window", "evidence_window_tokens": 384},
    {"evidence_mode": "window", "evidence_window_tokens": 512},
    {"evidence_mode": "full_parent"},
]


def _score_rows(dataset_dir: Path, rows: list[dict], split: str) -> dict:
    """把内存中的 run 行写入临时文件评分；扫描 run 不落正式 runs/ 目录。"""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        write_jsonl(tmp_path, rows)
        return score_run(dataset_dir, tmp_path, split=split)
    finally:
        tmp_path.unlink(missing_ok=True)


def _retrieval_metrics(dataset_dir, stack, queries, overrides, split) -> dict:
    import time

    meta = run_meta(dataset_dir, stack.store.index_name, "sweep", overrides, stage="retrieval")
    rows = []
    latencies = []
    for query in queries:
        started = time.perf_counter()
        try:
            documents, trace = stack.router.retrieve_detailed(query["question"], overrides)
            latency = (time.perf_counter() - started) * 1000.0
            latencies.append(latency)
            rows.append(retrieval_run_row(query, documents, latency, trace, meta))
        except Exception as exc:  # noqa: BLE001
            rows.append(error_run_row(query, str(exc), meta))
    score = _score_rows(dataset_dir, rows, split)
    return {**score["groups"]["overall"]["metrics"], "queries": len(rows)}


def _selection_key(metrics: dict) -> tuple:
    """锁定排序键：recall@10 > mrr@10 > ndcg@10；同分保持先出现顺序。"""
    return (
        float(metrics.get("recall@10", 0.0)),
        float(metrics.get("mrr@10", 0.0)),
        float(metrics.get("ndcg@10", 0.0)),
    )


def sweep_retrieval(
    dataset_dir: Path,
    stack,
    split: str = "dev",
    axes: dict[str, list[dict]] | None = None,
) -> dict:
    """基线 + 单变量扫描；返回带选择轨迹的结果（不写正式 run 文件）。"""
    queries = load_queries(dataset_dir, split)
    baselines = {
        name: _retrieval_metrics(dataset_dir, stack, queries, overrides, split)
        for name, overrides in DEFAULT_RETRIEVAL_VARIANTS.items()
    }
    selection: list[dict] = []
    picks: dict[str, dict] = {}
    for axis, combos in (axes or SWEEP_AXES).items():
        evaluated = []
        for overrides in combos:
            metrics = _retrieval_metrics(dataset_dir, stack, queries, overrides, split)
            evaluated.append({"overrides": overrides, "metrics": metrics})
        best = max(evaluated, key=lambda item: _selection_key(item["metrics"]))
        picks[axis] = best["overrides"]
        selection.append({"axis": axis, "variants": evaluated, "chosen": best["overrides"]})
    return {"baselines": baselines, "selection": selection, "picks": picks}


def decide_rerank(baselines: dict[str, dict]) -> dict:
    """Rerank 生效规则：recall@10 增益 >= 1pp 且 mrr@10 不下降才默认开启。"""
    hybrid = baselines.get("hybrid", {})
    rerank = baselines.get("hybrid_rerank", {})
    recall_gain = float(rerank.get("recall@10", 0.0)) - float(hybrid.get("recall@10", 0.0))
    mrr_gain = float(rerank.get("mrr@10", 0.0)) - float(hybrid.get("mrr@10", 0.0))
    enabled = recall_gain >= RERANK_MIN_RECALL_GAIN and mrr_gain >= 0
    return {
        "enabled": enabled,
        "recall@10_gain": round(recall_gain, 6),
        "mrr@10_gain": round(mrr_gain, 6),
        "rule": (
            f"recall@10 增益 >= {RERANK_MIN_RECALL_GAIN} 且 mrr@10 不下降才默认开启；"
            "无稳定收益保持默认关闭（算法层QA §7.2 P0-B）"
        ),
    }


def compose_locked_config(base: dict, picks: dict[str, dict], rerank_enabled: bool) -> dict:
    """合并各轴最优值；rerank 决策为关闭时剥离 rerank 覆盖，尊重生产默认。

    各轴独立选择可能出现 rerank_top_n > candidate_top_k 一类越界组合，
    在此把送排池钳制回 [top_k, candidate_top_k]，而不是把异常留给 Test 阶段。
    """
    from backend.src.config.retrieval_config import build_retrieval_config

    locked = dict(base)
    for axis in sorted(picks):
        locked.update(picks[axis])
    if not rerank_enabled:
        locked.pop("rerank_enabled", None)
        locked.pop("rerank_backend", None)
        locked.pop("rerank_top_n", None)
    if "rerank_top_n" in locked:
        top_k = int(locked.get("top_k", 5))
        candidate_top_k = int(locked.get("candidate_top_k", 30))
        locked["rerank_top_n"] = min(max(int(locked["rerank_top_n"]), top_k), candidate_top_k)
    build_retrieval_config(locked)
    return locked


def sweep_evidence(dataset_dir: Path, stack, split: str = "dev") -> dict:
    """证据层扫描（调用 Chat，仅 Dev）：模式 + 窗口 token。"""
    queries = load_queries(dataset_dir, split)
    evaluated = []
    for qa_overrides in EVIDENCE_SWEEP:
        meta = run_meta(dataset_dir, stack.store.index_name, "sweep_evidence", {}, qa_overrides, stage="qa")
        rows = []
        for query in queries:
            try:
                payload = stack.qa.query(query["question"], qa_config=qa_overrides)
                rows.append(qa_run_row(query, payload, meta))
            except Exception as exc:  # noqa: BLE001
                from evaluation.runner.production_adapter import NO_EVIDENCE_CODES

                code = getattr(exc, "code", "") or ""
                status = "no_evidence" if code in NO_EVIDENCE_CODES else "error"
                rows.append(error_run_row(query, str(exc), meta, status))
        score = _score_rows(dataset_dir, rows, split)
        overall = score["groups"]["overall"]
        evaluated.append(
            {
                "qa_config": qa_overrides,
                "metrics": overall["metrics"],
                "p95_total_ms": (overall["latency_ms"].get("total") or {}).get("p95"),
                # 真实用量汇总（scorer 只聚合数值字段），供成本核算
                "usage": overall.get("usage", {}),
            }
        )

    def evidence_key(item: dict) -> tuple:
        metrics = item["metrics"]
        p95 = item.get("p95_total_ms")
        return (
            float(metrics.get("evidence_recall", 0.0)),
            float(metrics.get("citation_validity", 0.0)),
            -(float(p95) if p95 is not None else float("inf")),
        )

    best = max(evaluated, key=evidence_key)
    return {"variants": evaluated, "chosen": best["qa_config"]}


def lock_and_report(
    dataset_dir: Path,
    stack,
    split: str = "dev",
    *,
    with_evidence: bool = True,
) -> dict:
    """扫描 → 锁定 → 写 locks/ 产物。返回锁定的完整配置。"""
    if split != "dev":
        raise ValueError("参数扫描只允许 dev 分片")
    sweep = sweep_retrieval(dataset_dir, stack, split)
    rerank_decision = decide_rerank(sweep["baselines"])
    locked_retrieval = compose_locked_config({}, sweep["picks"], rerank_decision["enabled"])

    lock = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "git_commit": git_commit(),
        "es_index": stack.store.index_name,
        "locked_retrieval_config": locked_retrieval,
        "rerank_decision": rerank_decision,
        "baselines": sweep["baselines"],
        "selection": sweep["selection"],
    }
    evidence_summary = None
    if with_evidence:
        evidence_summary = sweep_evidence(dataset_dir, stack, split)
        lock["locked_qa_config"] = evidence_summary["chosen"]
        lock["evidence_selection"] = evidence_summary["variants"]

    dirs = evaluation_dirs()
    dirs["locks"].mkdir(parents=True, exist_ok=True)
    lock_path = dirs["locks"] / "locked_config.json"
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    summary_path = dirs["locks"] / "sweep_summary.md"
    summary_path.write_text(_render_summary(lock), encoding="utf-8", newline="\n")
    return lock


def _render_summary(lock: dict) -> str:
    lines = [
        "# Dev 参数扫描与锁定摘要",
        "",
        f"- Split: `{lock['split']}`  Git: `{lock['git_commit']}`  ES index: `{lock['es_index']}`",
        f"- 锁定检索配置: `{json.dumps(lock['locked_retrieval_config'], ensure_ascii=False, sort_keys=True)}`",
        f"- Rerank 决策: {'开启' if lock['rerank_decision']['enabled'] else '保持默认关闭'}"
        f"（recall@10 增益 {lock['rerank_decision']['recall@10_gain']:.4f}，"
        f"mrr@10 增益 {lock['rerank_decision']['mrr@10_gain']:.4f}）",
        "",
        "## 基线（Dev）",
        "",
        "| 变体 | recall@10 | mrr@10 | ndcg@10 |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in lock["baselines"].items():
        lines.append(
            f"| {name} | {metrics.get('recall@10', 0):.4f} "
            f"| {metrics.get('mrr@10', 0):.4f} | {metrics.get('ndcg@10', 0):.4f} |"
        )
    lines.extend(["", "## 单变量扫描选择", ""])
    for entry in lock["selection"]:
        chosen = entry["chosen"]
        lines.append(f"### {entry['axis']} → `{json.dumps(chosen, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
        lines.append("| 配置 | recall@10 | mrr@10 | ndcg@10 |")
        lines.append("|---|---:|---:|---:|")
        for variant in entry["variants"]:
            metrics = variant["metrics"]
            lines.append(
                f"| `{json.dumps(variant['overrides'], ensure_ascii=False, sort_keys=True)}` "
                f"| {metrics.get('recall@10', 0):.4f} "
                f"| {metrics.get('mrr@10', 0):.4f} | {metrics.get('ndcg@10', 0):.4f} |"
            )
        lines.append("")
    if "locked_qa_config" in lock:
        lines.extend(
            [
                "## 证据模式（调用 Chat，仅 Dev）",
                "",
                f"- 锁定 QA 配置: `{json.dumps(lock['locked_qa_config'], ensure_ascii=False, sort_keys=True)}`",
                "",
                "| 配置 | evidence_recall | citation_validity | p95 total ms |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in lock["evidence_selection"]:
            p95 = variant.get("p95_total_ms")
            lines.append(
                f"| `{json.dumps(variant['qa_config'], ensure_ascii=False, sort_keys=True)}` "
                f"| {variant['metrics'].get('evidence_recall', 0):.4f} "
                f"| {variant['metrics'].get('citation_validity', 0):.4f} "
                f"| {p95 if p95 is not None else 'unavailable'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-name", type=str, required=True)
    parser.add_argument("--split", choices=("dev",), default="dev",
                        help="扫描只允许 Dev 分片")
    parser.add_argument("--skip-evidence", action="store_true", help="跳过证据层扫描（省 Chat 费用）")
    args = parser.parse_args()

    from evaluation.runner.production_adapter import build_stack

    stack = build_stack(args.index_name)
    lock = lock_and_report(args.dataset.resolve(), stack, args.split, with_evidence=not args.skip_evidence)
    print(f"锁定检索配置: {json.dumps(lock['locked_retrieval_config'], ensure_ascii=False, sort_keys=True)}")
    print(f"Rerank: {'开启' if lock['rerank_decision']['enabled'] else '保持默认关闭'}")
    if "locked_qa_config" in lock:
        print(f"锁定 QA 配置: {json.dumps(lock['locked_qa_config'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
