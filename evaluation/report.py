"""Render a deterministic Markdown report from score_run.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.4f}"


def render_report(score: dict) -> str:
    group_order = ["overall", "questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
    metric_order = [
        "hit@1", "hit@3", "hit@5", "hit@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
        "mrr@10", "ndcg@10", "evidence_recall", "evidence_precision",
        "citation_validity", "citation_document_precision",
    ]
    lines = [
        f"# {score['dataset_id']} 评测报告",
        "",
        f"- Split: `{score['split']}`",
        f"- Dataset manifest SHA256: `{score['dataset_manifest_sha256']}`",
        f"- Run SHA256: `{score['run_sha256']}`",
        "",
        "## 确定性指标",
        "",
        "| 分组 | Query | " + " | ".join(metric_order) + " |",
        "|---|---:|" + "---:|" * len(metric_order),
    ]
    for group in group_order:
        aggregate = score["groups"][group]
        values = [_format(aggregate["metrics"].get(metric)) for metric in metric_order]
        lines.append(f"| {group} | {aggregate['queries']} | " + " | ".join(values) + " |")
    lines.extend(["", "## 性能与用量", ""])
    overall = score["groups"]["overall"]
    if overall["latency_ms"]:
        lines.extend(["| 延迟字段 | 可用样本 | P50 (ms) | P95 (ms) |", "|---|---:|---:|---:|"])
        for field, values in overall["latency_ms"].items():
            lines.append(f"| {field} | {values['available']} | {_format(values['p50'])} | {_format(values['p95'])} |")
    else:
        lines.append("延迟：unavailable")
    lines.append("")
    if overall["usage"]:
        lines.extend(["| 用量字段 | 可用样本 | 总计 |", "|---|---:|---:|"])
        for field, values in overall["usage"].items():
            lines.append(f"| {field} | {values['available']} | {_format(values['total'])} |")
    else:
        lines.append("Token/成本：unavailable")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本报告只包含可由固定 qrels/run 确定性复算的指标。"
            "未运行 LLM Judge 时，不报告 Faithfulness 或 Answer Correctness。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    score = json.loads(args.score.read_text(encoding="utf-8"))
    report = render_report(score)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
