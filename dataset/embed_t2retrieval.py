"""Embed the local T2Retrieval corpus and queries with the configured model.

The script writes float32 NumPy memmaps plus ID/text metadata. It is resumable:
rerunning it skips rows recorded in the progress file. Use ``--limit`` for a
small API smoke test before starting the full corpus job.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.config.settings import settings
from backend.src.infrastructure.embedding_factory import build_embedding_model

LOGGER = logging.getLogger("t2retrieval_embedding")
DATASET_DIR = ROOT / "dataset" / "T2Retrieval"
OUTPUT_DIR = DATASET_DIR / "embeddings"


def _read_table(name: str):
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return pq.read_table(path)


def _input_text(row: dict, kind: str) -> str:
    if kind == "corpus":
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        return "\n".join(part for part in (title, text) if part)
    return str(row.get("text") or "").strip()


def _embed_split(name: str, model, batch_size: int, limit: int | None) -> None:
    table = _read_table(name)
    total = table.num_rows if limit is None else min(limit, table.num_rows)
    ids = [str(value) for value in table.column("_id")[:total].to_pylist()]
    texts = [_input_text(row, name) for row in table.slice(0, total).to_pylist()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT_DIR / name
    mmap_path = prefix.with_suffix(".f32.mmap")
    metadata_path = prefix.with_suffix(".metadata.json")
    progress_path = prefix.with_suffix(".progress.json")

    dimension = getattr(model, "dimensions", None)
    if dimension is None:
        raise ValueError("MVP_EMBEDDING_DIMENSIONS must be set for persistent memmap output")
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError(f"embedding dimension must be positive, got {dimension}")

    progress = {"next_row": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("total") != total or progress.get("dimension") != dimension:
            raise ValueError(f"existing progress does not match current {name} input")

    next_row = int(progress.get("next_row", 0))
    mode = "r+" if mmap_path.exists() else "w+"
    vectors = np.memmap(mmap_path, dtype="float32", mode=mode, shape=(total, dimension))
    if mode == "w+":
        metadata_path.write_text(
            json.dumps(
                {
                    "split": name,
                    "rows": total,
                    "dimension": dimension,
                    "model": model.model_name,
                    "backend": model.backend_name,
                    "ids_file": str(prefix.with_suffix(".ids.jsonl").name),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with prefix.with_suffix(".ids.jsonl").open("w", encoding="utf-8") as handle:
            for item_id in ids:
                handle.write(json.dumps({"id": item_id}, ensure_ascii=False) + "\n")

    progress.update({"total": total, "dimension": dimension, "model": model.model_name})
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    LOGGER.info("embedding %s rows=%d start=%d batch_size=%d", name, total, next_row, batch_size)
    try:
        for start in range(next_row, total, batch_size):
            end = min(start + batch_size, total)
            batch_vectors = model.encode(texts[start:end])
            if len(batch_vectors) != end - start:
                raise RuntimeError(f"embedding count mismatch for {name}: {start}:{end}")
            matrix = np.asarray(batch_vectors, dtype="float32")
            if matrix.shape != (end - start, dimension):
                raise RuntimeError(f"embedding shape mismatch: {matrix.shape} != {(end - start, dimension)}")
            vectors[start:end] = matrix
            vectors.flush()
            progress["next_row"] = end
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            LOGGER.info("embedded %s %d/%d", name, end, total)
    except Exception:
        # A first-batch failure leaves only an empty allocation and must not
        # block a later retry with a different --limit or corrected config.
        if int(progress.get("next_row", 0)) == 0:
            for path in (mmap_path, metadata_path, progress_path, prefix.with_suffix(".ids.jsonl")):
                path.unlink(missing_ok=True)
        raise

    vectors.flush()
    progress["complete"] = True
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    LOGGER.info("completed %s output=%s", name, mmap_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("corpus", "queries", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N rows")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = build_embedding_model()
    batch_size = args.batch_size or settings.integer("MVP_EMBEDDING_BATCH_SIZE", 16, positive=True)
    splits = ("corpus", "queries") if args.split == "both" else (args.split,)
    for split in splits:
        _embed_split(split, model, batch_size, args.limit)


if __name__ == "__main__":
    main()
