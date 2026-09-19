"""调用生产 IngestionPipeline 建独立评测索引，并生成 index manifest。

评测索引必须使用独立名称；CRUD-RAG 原始 doc_id 原样保留（否则无法与
qrels 对齐）。逐题/逐文档失败显式记录，不静默丢文档。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from evaluation import DATASET_ID
from evaluation.runner import RUNNER_VERSION, evaluation_dirs, git_commit
from evaluation.schemas import read_jsonl, sha256_file

from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.config.settings import settings

# 与生产 benchmark_adapter 相同的 html 嗅探规则（一行启发式，不是算法）
_HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")

DEFAULT_INDEX_NAME = f"rag-eval-{DATASET_ID}"


def sniff_file_type(text: str) -> str:
    return "html" if _HTML_RE.search(text) else "text"


def index_dataset(
    dataset_dir: Path,
    stack,  # ProductionStack
    *,
    chunk_config: ChunkConfig | None = None,
    limit: int | None = None,
    log_every: int = 100,
) -> dict:
    """把 corpus 全量写入 stack 指向的评测索引；返回 manifest 数据。"""
    corpus = [*read_jsonl(dataset_dir / "corpus.jsonl")]
    if limit is not None:
        corpus = corpus[:limit]
    chunk_config = chunk_config or ChunkConfig()

    indexed_docs = 0
    failed_docs: list[dict] = []
    record_count = 0
    for position, row in enumerate(corpus, start=1):
        doc_id = str(row["doc_id"])
        try:
            result = stack.pipeline.run(
                doc_id,
                {
                    "file_type": sniff_file_type(str(row.get("text", ""))),
                    "text": str(row.get("text", "")),
                    "doc_name": doc_id,
                },
                chunk_config,
            )
            indexed_docs += 1
            record_count += int(result.get("indexed_count", 0))
        except Exception as exc:  # noqa: BLE001 - 失败逐文档记录，不中断
            failed_docs.append({"doc_id": doc_id, "error": str(exc)[:300]})
        if log_every and position % log_every == 0:
            print(f"[index] {position}/{len(corpus)} indexed={indexed_docs} failed={len(failed_docs)}", flush=True)

    manifest = {
        "runner_version": RUNNER_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "git_commit": git_commit(),
        "es_index": stack.store.index_name,
        "chunk_profile_version": "parent-child-v2",
        "chunk_config": {
            "parent_target_tokens": chunk_config.parent_target_tokens,
            "parent_max_tokens": chunk_config.parent_max_tokens,
            "child_target_tokens": chunk_config.child_target_tokens,
            "child_max_tokens": chunk_config.child_max_tokens,
            "embedding_input_budget": chunk_config.embedding_input_budget,
        },
        "embedding_backend": settings.text("MVP_EMBEDDING_BACKEND", "glm"),
        "embedding_model": settings.text("MVP_EMBEDDING_MODEL", ""),
        "embedding_profile": "child-body-v1",
        "docs_total": len(corpus),
        "docs_indexed": indexed_docs,
        "docs_failed": len(failed_docs),
        "failed_documents": failed_docs,
        "records_indexed": record_count,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return manifest


def write_manifest(manifest: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-name", type=str, default=DEFAULT_INDEX_NAME)
    parser.add_argument("--limit", type=int, help="只索引前 N 篇（冒烟用）")
    parser.add_argument("--output", type=Path, help="manifest 输出路径（默认 evaluation/indexes/）")
    args = parser.parse_args()

    from evaluation.runner.production_adapter import build_stack

    stack = build_stack(str(args.index_name))
    manifest = index_dataset(args.dataset.resolve(), stack, limit=args.limit)
    output = args.output or evaluation_dirs()["indexes"] / f"{str(args.index_name)}.index_manifest.json"
    write_manifest(manifest, output)
    print(json.dumps({key: manifest[key] for key in (
        "es_index", "docs_total", "docs_indexed", "docs_failed", "records_indexed",
    )}, ensure_ascii=False))
    print(f"manifest 已写入 {output}")


if __name__ == "__main__":
    main()
