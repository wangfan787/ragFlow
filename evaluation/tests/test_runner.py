"""生产 Runner 契约（索引 manifest / run 录制 / 参数扫描锁定）。

- index_dataset 只调用生产 IngestionPipeline，CRUD doc_id 原样保留；
- run_retrieval / run_qa 产出符合冻结 schema 的 JSONL，可被 score_run 评分；
- 逐题错误如实记录（no_evidence / error），不静默丢题；
- sweep 只在 Dev 上单变量扫描并锁定配置，Rerank 无收益保持关闭。
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.schemas import read_jsonl
from evaluation.score_run import score_run


# ---------------------------------------------------------------------------
# 测试替身：与生产 ElasticsearchStore / Embeddings / ChatModel 契约一致
# ---------------------------------------------------------------------------

class FakeStore:
    """内存版检索存储：签名与返回结构与 ElasticsearchStore 保持一致。"""

    def __init__(self, index_name: str = "eval-fixture") -> None:
        self.index_name = index_name
        self.records: dict[str, object] = {}
        self.vectors: dict[str, list[float]] = {}

    def upsert(self, records, vectors) -> None:
        for record in records:
            self.records[record.metadata["chunk_id"]] = record
        self.vectors.update(vectors or {})

    def delete_stale_by_doc_id(self, doc_id: str, keep_ids: list[str]) -> None:
        for chunk_id in list(self.records):
            record = self.records[chunk_id]
            if record.metadata["doc_id"] == doc_id and chunk_id not in keep_ids:
                del self.records[chunk_id]

    @staticmethod
    def _matches(metadata: dict, filters: dict | None) -> bool:
        for field, expected in (filters or {}).items():
            value = metadata.get(field)
            if isinstance(expected, (list, tuple, set)):
                if value not in expected:
                    return False
            elif value != expected:
                return False
        return True

    def vector_search(self, query_vector, top_k, filters=None):
        scored = []
        for chunk_id, vector in self.vectors.items():
            record = self.records.get(chunk_id)
            if record is None or not self._matches(record.metadata, filters):
                continue
            dot = sum(a * b for a, b in zip(query_vector, vector))
            norm = math.sqrt(sum(v * v for v in query_vector) * sum(v * v for v in vector)) or 1.0
            scored.append(((1.0 + dot / norm) / 2.0, chunk_id, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        documents = []
        for score, _chunk_id, record in scored[:top_k]:
            documents.append(type(record)(
                page_content=record.page_content,
                metadata={**record.metadata, "score": score},
            ))
        return documents

    def keyword_search(self, query, top_k, filters=None):
        query_terms = set(re.findall(r"\w+", query))
        scored = []
        for chunk_id, record in self.records.items():
            if not self._matches(record.metadata, filters):
                continue
            text_terms = set(re.findall(
                r"\w+", record.page_content + " " + str(record.metadata.get("doc_name", "")),
            ))
            overlap = len(query_terms & text_terms)
            if overlap:
                scored.append((overlap, chunk_id, record))
        max_score = max((item[0] for item in scored), default=0)
        scored.sort(key=lambda item: (-item[0], item[1]))
        documents = []
        for raw, _chunk_id, record in scored[:top_k]:
            documents.append(type(record)(
                page_content=record.page_content,
                metadata={
                    **record.metadata,
                    "keyword_score": raw / max_score if max_score else 0.0,
                },
            ))
        return documents

    def query_by_ids(self, ids):
        return [self.records[chunk_id] for chunk_id in ids if chunk_id in self.records]


class FakeEmbedding:
    model = "fake-embed-v1"
    dimensions = 32

    @staticmethod
    def _vector(text: str) -> list[float]:
        values = [0.0] * 32
        for token in re.findall(r"\w+", text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            values[digest[0] % 32] += 1.0 if digest[1] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeChat:
    model_name = "fake-chat-v1"

    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(
            content="依据证据回答 [1]",
            usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
        )


# ---------------------------------------------------------------------------
# 使用 conftest.py 的共享合成数据；这里仅构造生产调用链
# ---------------------------------------------------------------------------

def _build_stack(index_name: str = "eval-fixture"):
    from evaluation.runner.production_adapter import build_stack

    store = FakeStore(index_name)
    return build_stack(
        index_name,
        embedding_model=FakeEmbedding(),
        chat_model=FakeChat(),
        store=store,
    )


@pytest.fixture()
def isolated_eval_dirs(tmp_path, monkeypatch):
    """把 runs/scores/reports/indexes/locks 全部指到临时目录。"""
    import evaluation.runner.run_qa as run_qa_module
    import evaluation.runner.run_retrieval as run_retrieval_module
    import evaluation.runner.sweep as sweep_module

    dirs = {
        "runs": tmp_path / "runs",
        "scores": tmp_path / "scores",
        "reports": tmp_path / "reports",
        "indexes": tmp_path / "indexes",
        "locks": tmp_path / "locks",
    }
    for module in (run_retrieval_module, run_qa_module, sweep_module):
        monkeypatch.setattr(module, "evaluation_dirs", lambda: dirs)
    return dirs


# ---------------------------------------------------------------------------
# index_dataset
# ---------------------------------------------------------------------------

def test_index_dataset_preserves_crud_doc_ids_and_writes_manifest(tmp_path: Path, monkeypatch, mini_dataset) -> None:
    from evaluation.runner import index_dataset as index_module
    from evaluation.runner.index_dataset import index_dataset, write_manifest

    monkeypatch.setattr(index_module, "git_commit", lambda: "a" * 40)
    dataset = mini_dataset
    stack = _build_stack()
    manifest = index_dataset(dataset, stack)

    corpus_ids = {row["doc_id"] for row in read_jsonl(dataset / "corpus.jsonl")}
    assert manifest["docs_total"] == 20
    assert manifest["docs_indexed"] == 20
    assert manifest["docs_failed"] == 0
    assert manifest["records_indexed"] > 20  # 父块+子块
    assert manifest["es_index"] == "eval-fixture"
    assert manifest["dataset_manifest_sha256"]
    assert manifest["git_commit"] == "a" * 40
    assert manifest["chunk_config"]["child_target_tokens"] > 0

    indexed_doc_ids = {record.metadata["doc_id"] for record in stack.store.records.values()}
    assert indexed_doc_ids == corpus_ids  # CRUD 原始 doc_id 必须原样保留


def test_index_dataset_records_per_document_failures(tmp_path: Path, mini_dataset) -> None:
    from evaluation.runner.index_dataset import index_dataset

    dataset = mini_dataset
    stack = _build_stack()

    original_run = stack.pipeline.run
    calls = {"count": 0}

    def flaky_run(doc_id, parse_config, chunk_config, *, owner_id=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise ValueError("boom")
        return original_run(doc_id, parse_config, chunk_config, owner_id=owner_id)

    stack.pipeline.run = flaky_run
    manifest = index_dataset(dataset, stack)
    assert manifest["docs_failed"] == 1
    assert manifest["failed_documents"][0]["error"].endswith("boom")


# ---------------------------------------------------------------------------
# run_retrieval / run_qa
# ---------------------------------------------------------------------------

def test_run_retrieval_records_valid_runs_for_all_variants(tmp_path: Path, isolated_eval_dirs, mini_dataset) -> None:
    from evaluation.runner.index_dataset import index_dataset
    from evaluation.runner.run_retrieval import DEFAULT_RETRIEVAL_VARIANTS, run_retrieval

    dataset = mini_dataset
    stack = _build_stack()
    index_dataset(dataset, stack)

    results = run_retrieval(dataset, stack, "dev", DEFAULT_RETRIEVAL_VARIANTS)
    assert {result["variant"] for result in results} == {
        "vector", "keyword", "hybrid", "hybrid_rerank",
    }
    for result in results:
        rows = [*read_jsonl(result["run_path"])]
        assert len(rows) == 3  # fixture dev split = 3 条
        assert all(row["status"] == "ok" for row in rows)
        for row in rows:
            ranks = [item["rank"] for item in row["retrieved"]]
            assert ranks == list(range(1, len(ranks) + 1))  # rank 连续从 1 开始
            meta = row["meta"]
            assert meta["variant"] == result["variant"]
            assert meta["es_index"] == "eval-fixture"
            assert len(meta["config_fingerprint"]) == 16
        assert result["score_path"].exists()
        assert result["report_path"].exists()


def test_run_retrieval_dedupes_documents_in_run_rows(tmp_path: Path) -> None:
    from evaluation.runner.production_adapter import retrieval_run_row
    from langchain_core.documents import Document

    documents = [
        Document(page_content="a", metadata={"doc_id": "d1", "score": 0.9}),
        Document(page_content="b", metadata={"doc_id": "d1", "score": 0.8}),
        Document(page_content="c", metadata={"doc_id": "d2", "score": 0.7}),
    ]
    row = retrieval_run_row({"query_id": "q", "question": "?"}, documents, 5.0, {"retrieval_mode": "hybrid"}, {})
    assert [item["doc_id"] for item in row["retrieved"]] == ["d1", "d2"]
    assert [item["rank"] for item in row["retrieved"]] == [1, 2]


def test_run_qa_records_answer_citations_and_evidence(tmp_path: Path, isolated_eval_dirs, mini_dataset) -> None:
    from evaluation.runner.index_dataset import index_dataset
    from evaluation.runner.run_qa import run_qa

    dataset = mini_dataset
    stack = _build_stack()
    index_dataset(dataset, stack)

    results = run_qa(dataset, stack, "dev")
    assert {result["variant"] for result in results} == {
        "hybrid_child_only", "hybrid_window", "hybrid_full_parent",
    }
    for result in results:
        rows = [*read_jsonl(result["run_path"])]
        assert len(rows) == 3
        for row in rows:
            assert row["status"] == "ok"
            assert row["answer"]
            assert row["citations"] and row["citations"][0]["citation_index"] == 1
            # usage 只含数值字段（"unavailable" 哨兵不得进入 run）
            assert row["usage"]["input_tokens"] == 50
            assert all(isinstance(value, (int, float)) for value in row["usage"].values())
            assert row["latency_ms"]["total"] > 0
            assert row["meta"]["qa_config"]["evidence_mode"]


def test_run_qa_maps_no_evidence_errors_per_query(tmp_path: Path, isolated_eval_dirs, mini_dataset) -> None:
    from evaluation.runner.run_qa import run_qa_variant

    dataset = mini_dataset
    stack = _build_stack()
    # 过滤到不存在的 doc：检索必然为空 → NO_RETRIEVED_CHUNKS → no_evidence
    spec = {"retrieval": {"filters": {"doc_id": "crud_nonexistent"}}, "qa": {}}
    result = run_qa_variant(
        dataset, stack, "dev", "empty", spec, runs_dir=isolated_eval_dirs["runs"],
        score_dir=isolated_eval_dirs["scores"], report_dir=isolated_eval_dirs["reports"],
    )
    rows = [*read_jsonl(result["run_path"])]
    assert len(rows) == 3
    assert all(row["status"] == "no_evidence" for row in rows)
    assert all(row["retrieved"] == [] and row["answer"] == "" for row in rows)
    # run 覆盖全部 query，scorer 不再报 incomplete
    assert result["metrics"]["hit@10"] == 0.0


# ---------------------------------------------------------------------------
# sweep 与锁定
# ---------------------------------------------------------------------------

def test_sweep_locks_config_and_applies_rerank_rule(tmp_path: Path, isolated_eval_dirs, mini_dataset) -> None:
    from evaluation.runner.index_dataset import index_dataset
    from evaluation.runner.sweep import (
        RERANK_MIN_RECALL_GAIN,
        decide_rerank,
        lock_and_report,
        sweep_retrieval,
    )

    dataset = mini_dataset
    stack = _build_stack()
    index_dataset(dataset, stack)

    sweep = sweep_retrieval(dataset, stack, "dev")
    # 基线四个变体齐全；每个轴的候选数量受轴定义约束（无笛卡尔积）
    assert set(sweep["baselines"]) == {"vector", "keyword", "hybrid", "hybrid_rerank"}
    total_variants = sum(len(entry["variants"]) for entry in sweep["selection"])
    assert total_variants == 4 + 3 + 4 + 6 + 3

    # rerank 决策规则可独立验证
    decision = decide_rerank({
        "hybrid": {"recall@10": 0.5, "mrr@10": 0.4},
        "hybrid_rerank": {"recall@10": 0.5 + RERANK_MIN_RECALL_GAIN, "mrr@10": 0.45},
    })
    assert decision["enabled"] is True
    decision = decide_rerank({
        "hybrid": {"recall@10": 0.5, "mrr@10": 0.4},
        "hybrid_rerank": {"recall@10": 0.505, "mrr@10": 0.4},
    })
    assert decision["enabled"] is False  # 增益不足 1pp → 保持默认关闭

    lock = lock_and_report(dataset, stack, "dev")
    locked = lock["locked_retrieval_config"]
    assert locked  # 锁定配置非空且能通过生产校验
    from backend.src.config.retrieval_config import build_retrieval_config
    build_retrieval_config(locked)
    assert "locked_qa_config" in lock and lock["locked_qa_config"]["evidence_mode"]
    assert (isolated_eval_dirs["locks"] / "locked_config.json").exists()
    summary = (isolated_eval_dirs["locks"] / "sweep_summary.md").read_text(encoding="utf-8")
    assert "Rerank 决策" in summary and "单变量扫描选择" in summary


def test_sweep_rejects_test_split(tmp_path: Path) -> None:
    from evaluation.runner.sweep import lock_and_report

    with pytest.raises(ValueError, match="dev"):
        lock_and_report(tmp_path, _build_stack(), "test", with_evidence=False)
