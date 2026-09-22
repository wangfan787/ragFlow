"""Parse → Chunk → Index → Retrieve 全链路演示（真实 GLM embedding + 本地 ES）。

数据源由 --source 选择：
  t2（默认）  T2Retrieval-subset 数据集：抽 1 条 query（默认 id=19）+
             词项重叠最高的若干干扰文档，检索结果对照 qrels 正例。
  demo       tests/parsing/data/sample_walkthrough.md 单篇示例文档，
             检索问题默认“死锁的四个必要条件是什么？”。

数据写入 demo 专属索引（默认 rag-demo-walkthrough，每次运行先删除重建），
不触碰产品索引。

前置条件：本地 Elasticsearch（默认 http://localhost:9200）已启动，
config/local.yaml 已配置 embedding.api_key / model / base_url。
命令行：python -m backend.scripts.demo.walkthrough_demo [--source t2|demo]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.chunking import ChunkConfig
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.models import build_embeddings
from backend.src.retrieval.hybrid_router import HybridRouter
from backend.src.retrieval.vectorizer import query_terms, tokenize

ROOT = Path(__file__).resolve().parents[3]
DEMO_MD = ROOT / "backend" / "tests" / "parsing" / "data" / "sample_walkthrough.md"
DEFAULT_INDEX = "rag-demo-walkthrough"
DEMO_QUERY = "死锁的四个必要条件是什么？"
_HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")

# 评测 Dev 扫描锁定的检索配置（2026-09-19，Test 集 recall@10 97.5%）；
# 注意生产代码 RetrievalConfig 的默认值仍是首版基线（top_k=5 /
# candidate_top_k=30 / threshold=0.1），尚未与锁定值同步。
# 想观察单个旋钮的效果直接改这里，或改 retrieval_mode 做 vector/keyword 消融。
RETRIEVAL_CONFIG: dict[str, Any] = {
    "retrieval_mode": "hybrid",
    "candidate_top_k": 10,
    "top_k": 10,
    "similarity_threshold": 0.2,
    "vector_weight": 0.75,
    "keyword_weight": 0.25,
    "rerank_enabled": False,
}


@dataclass(frozen=True)
class DemoCase:
    """一次演示的统一输入：检索 query、qrels 正例（demo 源为空）、待入库文档。"""

    source: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    documents: tuple[dict[str, str], ...]


def load_t2_case(
    query_id: str,
    distractor_count: int,
    dataset_dir: Path = ROOT / "dataset" / "T2Retrieval-subset",
) -> DemoCase:
    """抽 1 条 T2 query：qrels 正例文档 + 与 query 词项重叠最高的干扰文档。"""
    import pyarrow.parquet as pq

    if distractor_count < 0:
        raise ValueError("distractor_count must be >= 0")
    queries = pq.read_table(dataset_dir / "queries.parquet").to_pylist()
    row = next((row for row in queries if str(row["_id"]) == query_id), None)
    if row is None:
        raise ValueError(f"query_id={query_id} does not exist in {dataset_dir}")
    query = str(row["text"])
    qrels = pq.read_table(dataset_dir / "qrels.parquet").to_pylist()
    relevant = tuple(dict.fromkeys(
        str(row["corpus-id"]) for row in qrels
        if str(row["query-id"]) == query_id and float(row["score"]) > 0
    ))
    if not relevant:
        raise ValueError(f"query_id={query_id} has no positive qrels")
    corpus = pq.read_table(dataset_dir / "corpus.parquet").to_pylist()
    by_id = {str(row["_id"]): row for row in corpus}
    missing = set(relevant) - by_id.keys()
    if missing:
        raise ValueError(f"qrels refer to documents outside the corpus: {sorted(missing)}")
    terms = query_terms(query)
    distractors = sorted(
        (row for row in corpus if str(row["_id"]) not in relevant),
        key=lambda row: (
            -len(terms & set(tokenize(f"{row.get('title', '')} {row.get('text', '')}"))),
            str(row["_id"]),
        ),
    )
    selected = [by_id[doc_id] for doc_id in relevant] + distractors[:distractor_count]
    documents = []
    for row in selected:
        text = str(row.get("text") or "").strip()
        documents.append(
            {
                "doc_id": str(row["_id"]),
                "file_type": "html" if _HTML_RE.search(text) else "text",
                "text": text,
                "doc_name": str(row.get("title") or f"T2-{row['_id']}"),
            }
        )
    return DemoCase(
        source="t2",
        query=query,
        relevant_doc_ids=relevant,
        documents=tuple(documents),
    )


def load_demo_case(query: str | None) -> DemoCase:
    return DemoCase(
        source="demo",
        query=query or DEMO_QUERY,
        relevant_doc_ids=(),
        documents=(
            {
                "doc_id": "sample",
                "file_type": "md",
                "text": DEMO_MD.read_text(encoding="utf-8"),
                "doc_name": DEMO_MD.name,
            },
        ),
    )


def reset_demo_index(store: ElasticsearchStore) -> None:
    """演示数据只允许进 rag-demo-* 索引；每次运行清空重建，避免上次残留。"""
    if not store.index_name.startswith("rag-demo-"):
        raise ValueError(f"demo 只允许写 rag-demo-* 索引，当前：{store.index_name}")
    client = store.client
    if client.indices.exists(index=store.index_name):
        client.indices.delete(index=store.index_name)


def index_case(
    case: DemoCase,
    store: ElasticsearchStore,
    embeddings: Any,
) -> list[dict]:
    pipeline = IngestionPipeline(
        embedding_indexer=EmbeddingIndexer(embedding_model=embeddings, store=store)
    )
    trace = []
    for doc in case.documents:
        parsed = pipeline.parse_stage(
            doc["doc_id"],
            {"file_type": doc["file_type"], "text": doc["text"], "doc_name": doc["doc_name"]},
        )
        chunks = pipeline.chunk_stage(doc["doc_id"], parsed, ChunkConfig(), doc["doc_name"])
        indexed = pipeline.embedding_indexer.index(
            doc["doc_id"], chunks, doc_name=doc["doc_name"]
        )
        trace.append(
            {
                "doc_id": doc["doc_id"],
                "relevant": doc["doc_id"] in case.relevant_doc_ids,
                "file_type": doc["file_type"],
                "parsed_blocks": len(parsed),
                "parents": sum(c.metadata["chunk_role"] == "parent" for c in chunks),
                "children": sum(c.metadata["chunk_role"] == "child" for c in chunks),
                "indexed_records": indexed,
            }
        )
    return trace


def print_report(
    case: DemoCase,
    indexing_trace: list[dict],
    documents: list,
    trace: dict,
) -> None:
    print(f"\n=== 1. 数据源与检索 query（source={case.source}）===")
    print(f"query: {case.query}")
    if case.relevant_doc_ids:
        print(f"qrels relevant_doc_ids: {list(case.relevant_doc_ids)}")
    else:
        print("qrels: 无（单篇示例文档，只看检索行为）")

    print("\n=== 2. 真实入库（GLM embedding → ES）===")
    for row in indexing_trace:
        marker = "正例" if row["relevant"] else "干扰" if case.relevant_doc_ids else "文档"
        print(
            f"[{marker}] doc={row['doc_id']} type={row['file_type']} "
            f"blocks={row['parsed_blocks']} parents={row['parents']} "
            f"children={row['children']} indexed={row['indexed_records']}"
        )

    print("\n=== 3. 检索漏斗 ===")
    print(
        f"vector={trace['vector_candidate_count']} "
        f"keyword={trace['keyword_candidate_count']} "
        f"fused={trace['fused_candidate_count']} "
        f"eligible={trace['eligible_candidate_count']} "
        f"final={trace['chunk_count']}"
    )
    print(f"parent_expansion: {trace['parent_expansion']}")
    if trace.get("filtered_chunks"):
        print(f"阈值过滤示例: {trace['filtered_chunks'][:3]}")

    print("\n=== 4. 最终 Top 结果 ===")
    relevant = set(case.relevant_doc_ids)
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        marker = (
            "命中 qrels"
            if str(metadata["doc_id"]) in relevant
            else ("干扰文档" if relevant else "示例文档")
        )
        section = "/".join(metadata.get("section_path") or [])
        print(
            f"#{rank} [{marker}] doc={metadata['doc_id']} "
            f"section={section} "
            f"score={metadata['score']:.4f} "
            f"vector={metadata['vector_score']:.4f} "
            f"keyword={metadata['keyword_score']:.4f}"
        )
        print(f"   {document.page_content.replace(chr(10), ' ')[:120]}")

    if relevant:
        retrieved = {str(document.metadata["doc_id"]) for document in documents}
        hits = sorted(relevant & retrieved)
        print(f"\nTop-{len(documents)} qrels hits: {hits} / {sorted(relevant)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["t2", "demo"], default="t2")
    parser.add_argument("--query-id", default="19", help="t2 源：抽取的 query id")
    parser.add_argument("--distractors", type=int, default=8, help="t2 源：干扰文档数")
    parser.add_argument("--query", help="demo 源：检索问题（默认内置示例问题）")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="demo 专属 ES 索引名")
    args = parser.parse_args()

    if args.source == "t2":
        case = load_t2_case(args.query_id, args.distractors)
    else:
        case = load_demo_case(args.query)

    store = ElasticsearchStore(index_name=args.index)
    embeddings = build_embeddings()
    reset_demo_index(store)
    indexing_trace = index_case(case, store, embeddings)

    router = HybridRouter(store=store, embedding_model=embeddings)
    documents, trace = router.retrieve_detailed(case.query, RETRIEVAL_CONFIG)
    print_report(case, indexing_trace, documents, trace)


if __name__ == "__main__":
    main()
