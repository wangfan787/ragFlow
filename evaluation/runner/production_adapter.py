"""生产组件组装与 run schema 映射（评测计划 §4.1）。

本模块是 Runner 与 backend 之间唯一的边界：
- build_stack 用生产构造器组装 IngestionPipeline / HybridRouter / QAService，
  并让三者共享同一个（评测专用）ES store；
- payload → run row 的映射只做字段搬运与 doc_id 去重，不实现任何算法。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from evaluation import DATASET_ID
from evaluation.runner import RUNNER_VERSION, config_fingerprint, git_commit
from evaluation.schemas import sha256_file

from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.apps.services.qa_service import QAService
from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.retrieval.hybrid_router import HybridRouter

# 与生产无证据/预算耗尽语义对应的 ServiceError code → run status
NO_EVIDENCE_CODES = {"NO_RETRIEVED_CHUNKS", "QA_CONTEXT_BUDGET_EXHAUSTED"}


@dataclass
class ProductionStack:
    """共享同一 store 的生产组件集合；Runner 不绕过它们拼实验实现。"""

    store: ElasticsearchStore
    indexer: Any  # EmbeddingIndexer
    pipeline: IngestionPipeline
    router: HybridRouter
    qa: QAService


def build_stack(
    index_name: str,
    *,
    embedding_model=None,
    chat_model=None,
    store: ElasticsearchStore | None = None,
) -> ProductionStack:
    """组装共享同一 store 的生产组件：索引、检索、问答同源。"""
    store = store or ElasticsearchStore(index_name=index_name)
    indexer = EmbeddingIndexer(embedding_model=embedding_model, store=store)
    pipeline = IngestionPipeline(embedding_indexer=indexer)
    router = HybridRouter(store=store, embedding_model=embedding_model)
    qa = QAService(model=chat_model)
    qa.retriever = router
    return ProductionStack(store=store, indexer=indexer, pipeline=pipeline, router=router, qa=qa)


def run_meta(
    dataset_dir,
    index_name: str,
    variant: str,
    retrieval_config: RetrievalConfig | dict | None,
    qa_config: dict | None = None,
    *,
    stage: str,
) -> dict:
    """run 文件级指纹：dataset / Git / ES index / config / model 全部落盘。"""
    config_obj = (
        retrieval_config if isinstance(retrieval_config, RetrievalConfig)
        else build_retrieval_config(retrieval_config or {})
    )
    fingerprint_source = {
        "retrieval_config": asdict(config_obj),
        "qa_config": qa_config or {},
    }
    return {
        "runner_version": RUNNER_VERSION,
        "stage": stage,
        "variant": variant,
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "git_commit": git_commit(),
        "es_index": index_name,
        "retrieval_config": asdict(config_obj),
        "qa_config": qa_config or {},
        "config_fingerprint": config_fingerprint(fingerprint_source),
    }


def dedupe_by_doc(documents: list[dict]) -> list[dict]:
    """run schema 不允许同一 doc 重复出现；同 doc 多块只保留最高名次。"""
    seen: set[str] = set()
    rows = []
    for item in documents:
        doc_id = str(item["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(item)
    return rows


def retrieval_run_row(
    query: dict,
    documents: list,  # HybridRouter 返回的 Document 列表
    latency_ms: float,
    trace: dict,
    meta: dict,
) -> dict:
    channel = trace.get("retrieval_mode")
    retrieved = []
    for item in dedupe_by_doc(
        [{"doc_id": doc.metadata["doc_id"], "score": doc.metadata.get("score")}
         for doc in documents]
    ):
        retrieved.append(
            {
                "doc_id": item["doc_id"],
                "rank": len(retrieved) + 1,
                "score": round(float(item["score"]), 6),
                "channel": channel,
            }
        )
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "retrieved": retrieved,
        # 检索 run 没有真实 Prompt；候选文档即"将进入证据池的文档"，如实标注
        "evidence_doc_ids": [row["doc_id"] for row in retrieved],
        "answer": "",
        "citations": [],
        "status": "ok",
        "latency_ms": {"retrieval": round(latency_ms, 3), "total": round(latency_ms, 3)},
        "usage": {},
        "meta": meta,
    }


def error_run_row(query: dict, error: str, meta: dict, status: str = "error") -> dict:
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "retrieved": [],
        "evidence_doc_ids": [],
        "answer": "",
        "citations": [],
        "status": status,
        "error": error[:300],
        "latency_ms": {},
        "usage": {},
        "meta": meta,
    }


def _numeric_usage(usage: dict) -> dict:
    """只保留数值字段；"unavailable" 哨兵不写入 run（scorer 拒绝非数值）。"""
    return {
        key: value for key, value in (usage or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def qa_run_row(query: dict, payload: dict, meta: dict) -> dict:
    trace = payload.get("trace", {})
    documents = [
        {"doc_id": chunk.get("doc_id"), "score": chunk.get("score")}
        for chunk in payload.get("retrieved_chunks", [])
    ]
    retrieved = []
    for item in dedupe_by_doc(documents):
        retrieved.append(
            {
                "doc_id": item["doc_id"],
                "rank": len(retrieved) + 1,
                "score": round(float(item["score"] or 0.0), 6),
                "channel": trace.get("retrieval_mode"),
            }
        )
    citations = [
        {
            "citation_index": citation["citation_index"],
            "doc_id": citation["doc_id"],
            "chunk_id": citation["chunk_id"],
            **({"asset_id": citation["asset_id"]} if citation.get("asset_id") else {}),
        }
        for citation in payload.get("citations", [])
    ]
    timings = trace.get("timings_ms", {})
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "retrieved": retrieved,
        # 实际进入 Prompt 的证据文档（QA budget trace 原样录制）
        "evidence_doc_ids": list(trace.get("qa_budget", {}).get("evidence_doc_ids", [])),
        "answer": payload.get("answer", ""),
        "citations": citations,
        "status": "ok",
        "latency_ms": {key: value for key, value in timings.items() if isinstance(value, (int, float))},
        "usage": _numeric_usage(trace.get("usage", {})),
        "meta": meta,
    }


def timed(fn, *args, **kwargs):
    """单调时钟计时包装；返回 (结果, 毫秒)。"""
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000.0
