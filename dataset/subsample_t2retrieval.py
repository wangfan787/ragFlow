"""Build a query-first T2Retrieval subset for cost-bounded embedding evaluation.

抽样策略：先抽 query，被抽 query 的全部正例文档必须进入候选池（保证指标不系统性偏低），
剩余池名额用随机干扰文档补齐。指标只在池内计算，报告时需标注 pool 规模。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "T2Retrieval"
DST_DIR = ROOT / "T2Retrieval-subset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--pool-docs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = pq.read_table(SRC_DIR / "corpus.parquet").to_pylist()
    queries = pq.read_table(SRC_DIR / "queries.parquet").to_pylist()
    qrels = pq.read_table(SRC_DIR / "qrels.parquet").to_pylist()

    positives_by_query: dict[str, set[str]] = {}
    for row in qrels:
        if int(row["score"]) > 0:
            positives_by_query.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    eligible = [row for row in queries if positives_by_query.get(str(row["_id"]))]
    if args.num_queries > len(eligible):
        raise SystemExit(f"可抽 query 只有 {len(eligible)} 条，少于 --num-queries={args.num_queries}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(eligible, args.num_queries), key=lambda row: int(row["_id"]))
    sampled_ids = {str(row["_id"]) for row in sampled}

    pool: set[str] = set()
    for qid in sampled_ids:
        pool |= positives_by_query[qid]
    if len(pool) > args.pool_docs:
        raise SystemExit(
            f"被抽 query 的正例已有 {len(pool)} 篇，超过 --pool-docs={args.pool_docs}；请减少 query 数"
        )
    candidates = [str(row["_id"]) for row in corpus if str(row["_id"]) not in pool]
    pool |= set(rng.sample(candidates, args.pool_docs - len(pool)))

    corpus_rows = [row for row in corpus if str(row["_id"]) in pool]
    qrels_rows = [row for row in qrels if str(row["query-id"]) in sampled_ids]

    DST_DIR.mkdir(parents=True, exist_ok=True)
    subset = {
        "corpus.parquet": corpus_rows,
        "queries.parquet": sampled,
        "qrels.parquet": qrels_rows,
    }
    for name, rows in subset.items():
        pq.write_table(py_table(rows), DST_DIR / name)

    manifest = {
        "seed": args.seed,
        "num_queries": len(sampled),
        "pool_docs": len(pool),
        "qrels_pairs": len(qrels_rows),
        "source_sha256": {
            name: _sha256(SRC_DIR / name)
            for name in ("corpus.parquet", "queries.parquet", "qrels.parquet")
        },
        "source_sizes": {
            name: (SRC_DIR / name).stat().st_size
            for name in ("corpus.parquet", "queries.parquet", "qrels.parquet")
        },
    }
    (DST_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def py_table(rows: list[dict]):
    import pyarrow as pa

    return pa.table({key: [row[key] for row in rows] for key in rows[0]})


if __name__ == "__main__":
    main()
