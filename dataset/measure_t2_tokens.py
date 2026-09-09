"""Measure exact embedding-token cost of a T2Retrieval corpus through production components.

复用生产 Parser/Chunker/EmbeddingTextBuilder，逐 Child 累加 embedding_input_tokens，
得出“如果用 GLM 正式入库这条语料，需要上传多少 token”的口径一致估计（cl100k 计数）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus", nargs="?", default=None,
        help="corpus.parquet 路径；默认 dataset/T2Retrieval-subset/corpus.parquet",
    )
    args = parser.parse_args()
    path = Path(args.corpus) if args.corpus else ROOT / "dataset" / "T2Retrieval-subset" / "corpus.parquet"

    adapter = ProductionRagAdapter()
    rows = pq.read_table(path).to_pylist()

    total_tokens = 0
    total_children = 0
    per_doc_tokens: list[int] = []
    skipped: list[str] = []
    for index, row in enumerate(rows):
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        doc_id = str(row["_id"])
        try:
            children = adapter.child_inputs(doc_id=doc_id, doc_name=title, text=text)
        except Exception as exc:  # 单条失败不应中断测量；入库时同样会失败，需要单独归因
            skipped.append(f"{doc_id}: {type(exc).__name__}: {exc}")
            continue
        doc_tokens = sum(child["embedding_input_tokens"] for child in children)
        total_tokens += doc_tokens
        total_children += len(children)
        per_doc_tokens.append(doc_tokens)
        if (index + 1) % 1000 == 0:
            print(f"progress {index + 1}/{len(rows)} tokens={total_tokens}", file=sys.stderr)

    per_doc_tokens.sort()
    rank = lambda q: per_doc_tokens[min(len(per_doc_tokens) - 1, int(q * len(per_doc_tokens)))]
    report = {
        "corpus": str(path),
        "docs_measured": len(per_doc_tokens),
        "docs_skipped": len(skipped),
        "children_total": total_children,
        "children_per_doc_mean": round(total_children / max(1, len(per_doc_tokens)), 2),
        "embedding_tokens_total": total_tokens,
        "embedding_tokens_per_doc": {
            "mean": round(statistics.fmean(per_doc_tokens), 1),
            "p50": rank(0.50),
            "p90": rank(0.90),
            "p95": rank(0.95),
            "p99": rank(0.99),
        },
        "skip_examples": skipped[:5],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
