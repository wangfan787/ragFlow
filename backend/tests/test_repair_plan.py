from __future__ import annotations

import json
import hashlib
from pathlib import Path

from backend.src.apps.services.qa_service import QAService
from backend.src.chunking import ChunkConfig, MarkdownChunker
from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.contracts import VectorRecord
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.parsing.parser_factory import build_parser
from backend.src.retrieval.hybrid_router import HybridRouter
from backend.src.retrieval.models import RetrievedChunk


class FakeEmbedding:
    backend_name = "fake"
    model_name = "fake-v1"
    dimensions = 2
    max_input_tokens = None

    def __init__(self):
        self.inputs: list[str] = []

    def encode(self, texts):
        self.inputs.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, records=None):
        self.records = records or []
        self.upserted = []

    def delete_by_doc_id(self, doc_id):
        return None

    def delete_stale_by_doc_id(self, doc_id, keep_ids):
        return None

    def upsert(self, records):
        self.upserted = records

    def query_by_ids(self, ids):
        return [record for record in self.records if record.id in ids]

    def vector_search(self, query_vector, top_k, filters=None):
        return []

    def keyword_search(self, query, top_k, filters=None):
        return []


def _block(text: str, block_type: str = "paragraph") -> dict:
    return {
        "doc_id": "doc",
        "text": text,
        "block_type": block_type,
        "section_path": ["section"],
        "order": 1,
        "source_span": {
            "start_line": 1,
            "end_line": 1,
            "start_char": 0,
            "end_char": len(text),
            "accuracy": "exact",
        },
    }


def test_strict_chunking_conserves_unpunctuated_code_and_table():
    counter = SimpleTokenCounter()
    config = ChunkConfig(
        parent_target_tokens=40,
        parent_max_tokens=50,
        child_target_tokens=12,
        child_max_tokens=16,
        embedding_input_budget=64,
    )
    for block_type, text in (
        ("paragraph", "无标点中文" * 100),
        ("code", "identifier_without_spaces=" + "x" * 500),
        ("table", "|cell|" * 300),
    ):
        chunks = MarkdownChunker().chunk([_block(text, block_type)], config)
        parents = [row for row in chunks if row["chunk_role"] == "parent"]
        children = [row for row in chunks if row["chunk_role"] == "child"]
        assert parents and children
        assert all(counter.count(row["text"]) <= config.parent_max_tokens for row in parents)
        assert all(counter.count(row["text"]) <= config.child_max_tokens for row in children)
        assert all(not row["retrieval_eligible"] for row in parents)
        assert all(row["retrieval_eligible"] for row in children)
        for parent in parents:
            family = [row for row in children if row["parent_id"] == parent["chunk_id"]]
            assert "".join(row["text"] for row in family) == parent["text"]


def test_child_serialization_preserves_boundary_whitespace_and_exact_span():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    config = ChunkConfig(
        parent_target_tokens=40,
        parent_max_tokens=50,
        child_target_tokens=3,
        child_max_tokens=4,
        embedding_input_budget=50,
    )
    chunks = MarkdownChunker().chunk([_block(text)], config)
    parent = next(row for row in chunks if row["chunk_role"] == "parent")
    children = [row for row in chunks if row["parent_id"] == parent["chunk_id"]]
    assert "".join(row["text"] for row in children) == parent["text"] == text
    for child in children:
        span = child["source_span"]
        assert text[span["start_char"] : span["end_char"]] == child["text"]


def test_text_and_html_parsers_accept_in_memory_sources(tmp_path: Path):
    text_blocks = build_parser("text").parse(
        "text-doc", {"file_type": "text", "text": "first\n\nsecond", "doc_name": "x.txt"}
    )
    assert [row["text"] for row in text_blocks] == ["first", "second"]
    assert text_blocks[1]["source_span"]["accuracy"] == "exact"

    html_blocks = build_parser("html").parse(
        "html-doc",
        {
            "file_type": "html",
            "text": "<h1>Title</h1><script>bad()</script><p>Hello &amp; world</p>",
            "doc_name": "x.html",
        },
    )
    assert [row["block_type"] for row in html_blocks] == ["heading", "paragraph"]
    assert "bad" not in " ".join(row["text"] for row in html_blocks)
    assert html_blocks[1]["source_span"]["accuracy"] == "line_only"
    loose_blocks = build_parser("html").parse(
        "loose-html",
        {"file_type": "html", "text": "before<div><p>inside</p>after</div>", "doc_name": "x"},
    )
    assert [row["text"] for row in loose_blocks] == ["before", "inside", "after"]
    repeated = build_parser("html").parse(
        "repeat", {"file_type": "html", "text": "<p>same</p>\n<p>same</p>", "doc_name": "x"}
    )
    assert repeated[0]["source_span"]["start_char"] == 0
    assert repeated[1]["source_span"]["start_char"] > repeated[0]["source_span"]["end_char"]


def test_markdown_preserves_indented_code_and_html_blocks(tmp_path: Path):
    source = tmp_path / "mixed.md"
    source.write_text(
        "before\n\n    secret_code()\n\n<div>secret html</div>\n\nafter",
        encoding="utf-8",
    )
    blocks = build_parser("md").parse(
        "mixed", {"file_type": "md", "file_path": str(source), "doc_name": source.name}
    )
    content = "\n".join(row["text"] for row in blocks)
    assert "secret_code()" in content
    assert "secret html" in content


def test_markdown_preserves_nested_code_and_pipe_less_gfm_table(tmp_path: Path):
    source = tmp_path / "containers.md"
    source.write_text(
        "- item\n\n  ```python\n  secret_list()\n  ```\n\n"
        "> before\n>\n>     secret_quote()\n\n"
        "Name | Value\n--- | ---\nalpha | secret_table\n\n"
        "<hr>\n\nafter",
        encoding="utf-8",
    )
    blocks = build_parser("md").parse(
        "containers",
        {"file_type": "md", "file_path": str(source), "doc_name": source.name},
    )
    content = "\n".join(row["text"] for row in blocks)
    assert "secret_list()" in content
    assert "secret_quote()" in content
    assert "secret_table" in content
    assert "after" in content


def test_file_parser_rejects_invalid_utf8_instead_of_dropping_bytes(tmp_path: Path):
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"before\xffafter")
    try:
        build_parser("text").parse(
            "invalid", {"file_type": "text", "file_path": str(source), "doc_name": source.name}
        )
    except ValueError as exc:
        assert "UTF-8" in str(exc)
    else:
        raise AssertionError("invalid UTF-8 must not be silently ignored")


def test_indexer_embeds_only_children_and_keeps_parent_context_record():
    chunks = MarkdownChunker().chunk([_block("正文内容。" * 80)], ChunkConfig())
    for chunk in chunks:
        chunk["doc_name"] = "doc.md"
        chunk["question_kwd"] = ["问题"]
    embedding = FakeEmbedding()
    store = FakeStore()
    count = EmbeddingIndexer(embedding_model=embedding, store=store).index("doc", chunks, "doc.md")
    children = [row for row in chunks if row["retrieval_eligible"]]
    assert count == len(chunks)
    assert len(embedding.inputs) == len(children)
    assert all("Content:" in text and "正文内容" in text for text in embedding.inputs)
    assert all(record.vector is None for record in store.upserted if not record.payload["retrieval_eligible"])
    assert all(record.vector for record in store.upserted if record.payload["retrieval_eligible"])


def test_indexer_uses_model_input_limit_before_adapter_guard():
    chunks = MarkdownChunker().chunk([_block("content " * 60)], ChunkConfig())
    for chunk in chunks:
        chunk.update({"doc_name": "very-long-name" * 20, "question_kwd": ["question " * 100]})
    embedding = FakeEmbedding()
    embedding.max_input_tokens = 64
    store = FakeStore()
    EmbeddingIndexer(embedding_model=embedding, store=store).index("doc", chunks, "doc")
    assert all(SimpleTokenCounter().count(text) <= 64 for text in embedding.inputs)


def test_family_mean_returns_parent_and_all_contributing_children():
    parent = VectorRecord(
        id="parent",
        doc_id="doc",
        vector=None,
        payload={
            "chunk_id": "parent",
            "doc_id": "doc",
            "content": "parent context",
            "chunk_role": "parent",
            "retrieval_eligible": False,
            "child_ids": ["c1", "c2"],
            "source_span": {"start_char": 0, "end_char": 14, "accuracy": "exact"},
        },
    )
    router = HybridRouter(store=FakeStore([parent]), embedding_model=FakeEmbedding())
    rows, trace = router._expand_parent_context(
        [
            {
                "chunk_id": "c1", "doc_id": "doc", "content": "one", "chunk_role": "child",
                "retrieval_eligible": True, "parent_id": "parent", "score": 0.9,
                "fused_score": 0.9, "vector_score": 0.8, "keyword_score": 1.0,
                "parent_char_start": 0, "parent_char_end": 3,
            },
            {
                "chunk_id": "c2", "doc_id": "doc", "content": "two", "chunk_role": "child",
                "retrieval_eligible": True, "parent_id": "parent", "score": 0.5,
                "fused_score": 0.5, "vector_score": 0.4, "keyword_score": 0.6,
                "parent_char_start": 4, "parent_char_end": 7,
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "parent"
    assert rows[0]["score"] == 0.7
    assert [item["chunk_id"] for item in rows[0]["matched_children"]] == ["c1", "c2"]
    assert trace["family_score"] == "mean"


def test_missing_parent_exposes_only_prompt_available_child_but_keeps_family_trace():
    router = HybridRouter(store=FakeStore(), embedding_model=FakeEmbedding())
    rows, _ = router._expand_parent_context(
        [
            {"chunk_id": "c1", "doc_id": "d", "content": "first", "chunk_role": "child",
             "retrieval_eligible": True, "parent_id": "missing", "score": 0.8,
             "fused_score": 0.8, "vector_score": 0.8, "keyword_score": 0.0},
            {"chunk_id": "c2", "doc_id": "d", "content": "second", "chunk_role": "child",
             "retrieval_eligible": True, "parent_id": "missing", "score": 0.6,
             "fused_score": 0.6, "vector_score": 0.6, "keyword_score": 0.0},
        ]
    )
    assert [item["chunk_id"] for item in rows[0]["matched_children"]] == ["c1"]
    assert [item["chunk_id"] for item in rows[0]["family_contributors"]] == ["c1", "c2"]


class FakeEsClient:
    def __init__(self):
        self.requests = []

    def search(self, **kwargs):
        self.requests.append(kwargs)
        return {"hits": {"hits": []}}


def test_store_adds_unoverrideable_child_filter_to_knn_and_bm25():
    client = FakeEsClient()
    store = ElasticsearchStore(client=client, index_name="v2")
    store.vector_search([1.0, 0.0], 5, {"retrieval_eligible": False, "doc_id": "doc"})
    store.keyword_search("query", 5, {"retrieval_eligible": False, "doc_id": "doc"})
    knn_filters = client.requests[0]["knn"]["filter"]
    keyword_filters = client.requests[1]["query"]["bool"]["filter"]
    assert {"term": {"retrieval_eligible": True}} in knn_filters
    assert {"term": {"retrieval_eligible": True}} in keyword_filters


class FakeAnswerModel:
    model_name = "fake-chat"
    context_limit_tokens = 1200
    completion_reserve_tokens = 200

    def __init__(self):
        self.counter = SimpleTokenCounter()
        self.messages = None

    def count_tokens(self, messages):
        return 2 + sum(4 + self.counter.count(item["content"]) for item in messages)

    def complete(self, messages):
        self.messages = messages
        return "answer [1]"


class FakeRouter:
    last_trace = {}

    def __init__(self, chunk):
        self.chunk = chunk

    def retrieve(self, question, retrieval_config=None):
        return [self.chunk]


def test_qa_window_is_anchored_near_matched_child_and_citation_uses_prompt_window():
    prefix = "前文" * 2000
    child_text = "真正命中的尾部证据"
    parent_text = prefix + child_text + "后文" * 200
    start = len(prefix)
    chunk = RetrievedChunk(
        chunk_id="parent", doc_id="doc", content=parent_text, score=0.8,
        vector_score=0.8, keyword_score=0.0, fused_score=0.8, rerank_score=None,
        section_path=["tail"], page_no=None, chunk_role="parent",
        matched_child_id="child", primary_matched_child_id="child",
        matched_children=[{
            "chunk_id": "child", "score": 0.8, "snippet": child_text,
            "parent_char_start": start, "parent_char_end": start + len(child_text),
            "source_span": {"start_char": start, "end_char": start + len(child_text)},
        }],
    )
    model = FakeAnswerModel()
    service = QAService(answer_llm=model)
    service.evidence_window_tokens = 200
    service.retriever = FakeRouter(chunk)
    payload = service.query("尾部是什么")
    assert child_text in model.messages[1]["content"]
    assert payload["citations"][0]["primary_matched_child_id"] == "child"
    assert child_text in payload["citations"][0]["context_snippet"]
    assert payload["trace"]["qa_budget"]["prompt_used_tokens"] <= 744


def test_qa_builds_separate_windows_for_distant_contributing_children():
    text = "A" * 500 + "first-hit" + "B" * 2000 + "second-hit" + "C" * 500
    first = text.index("first-hit")
    second = text.index("second-hit")
    chunk = RetrievedChunk(
        chunk_id="parent", doc_id="doc", content=text, score=0.8,
        vector_score=0.8, keyword_score=0.0, fused_score=0.8, rerank_score=None,
        section_path=[], page_no=None, chunk_role="parent", matched_child_id="c1",
        primary_matched_child_id="c1", matched_children=[
            {"chunk_id": "c1", "snippet": "first-hit", "parent_char_start": first, "parent_char_end": first + 9},
            {"chunk_id": "c2", "snippet": "second-hit", "parent_char_start": second, "parent_char_end": second + 10},
        ],
    )
    service = QAService(answer_llm=FakeAnswerModel())
    windows = service.window_builder.candidates([chunk], top_k=1, max_window_tokens=40)
    assert len(windows) == 2
    assert "first-hit" in windows[0].content
    assert "second-hit" in windows[1].content


def test_qa_second_pass_shrink_keeps_parent_absolute_anchor():
    text = "前文" * 2000 + "命中证据XYZ" + "后文" * 1000
    start = text.index("命中证据XYZ")
    chunk = RetrievedChunk(
        chunk_id="p", doc_id="d", content=text, score=1.0, vector_score=1.0,
        keyword_score=0.0, fused_score=1.0, rerank_score=None, section_path=[], page_no=None,
        matched_child_id="c", primary_matched_child_id="c", matched_children=[{
            "chunk_id": "c", "snippet": "命中证据XYZ", "parent_char_start": start,
            "parent_char_end": start + len("命中证据XYZ"),
        }],
    )
    builder = QAService(answer_llm=FakeAnswerModel()).window_builder
    first = builder.anchored(chunk, 200)
    second = builder.anchored(first, 50)
    assert "命中证据XYZ" in second.content
    assert second.prompt_span["parent_char_start"] <= start < second.prompt_span["parent_char_end"]


def test_qa_missing_coordinates_falls_back_to_child_without_snippet_guessing():
    text = "TARGET" + "middle" * 500 + "TARGET"
    chunk = RetrievedChunk(
        chunk_id="p", doc_id="d", content=text, score=1.0, vector_score=1.0,
        keyword_score=0.0, fused_score=1.0, rerank_score=None, section_path=[], page_no=None,
        matched_child_id="c", primary_matched_child_id="c",
        matched_children=[{"chunk_id": "c", "snippet": "TARGET"}],
    )
    window = QAService(answer_llm=FakeAnswerModel()).window_builder.anchored(chunk, 10)
    assert window.content == "TARGET"
    assert window.prompt_span == {"fallback": "matched_child"}


def test_t2_preparation_emits_v2_chunk_mapping_with_fingerprint(tmp_path, monkeypatch):
    from dataset import embed_t2retrieval
    from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter

    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path)
    manifest, progress = embed_t2retrieval._prepare_manifest(
        "corpus", ProductionRagAdapter(), 2
    )
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert progress["complete"] is True
    assert progress["schema_version"] == "t2-production-rag-v2"
    assert records and all(row["chunk_id"] and row["doc_id"] for row in records)
    assert all(row["embedding_text"] for row in records)


def test_rebuild_preflight_refuses_missing_registered_sources(tmp_path, monkeypatch):
    from backend.scripts import rebuild_v2_index

    missing = tmp_path / "missing.md"
    monkeypatch.setattr(
        rebuild_v2_index,
        "list_documents",
        lambda: [{"doc_id": "d", "file_path": str(missing), "file_type": "md", "name": "d"}],
    )
    try:
        rebuild_v2_index.rebuild("definitely-v2", dry_run=True)
    except FileNotFoundError as exc:
        assert "d" in str(exc)
    else:
        raise AssertionError("missing source must fail v2 preflight")


def test_t2_family_evaluator_means_hit_children_before_document_ranking():
    import numpy as np
    from dataset.evaluate_t2retrieval import _family_document_ranking

    mapping = [
        {"chunk_id": "a1", "parent_id": "pa", "doc_id": "A"},
        {"chunk_id": "a2", "parent_id": "pa", "doc_id": "A"},
        {"chunk_id": "b1", "parent_id": "pb", "doc_id": "B"},
    ]
    # A has the highest individual Child but family mean .5; B wins with .7.
    ranked = _family_document_ranking(
        np.array([0, 1, 2]),
        np.array([0.9, 0.1, 0.7]),
        mapping,
        aggregation="max",
        top_n=3,
        top_k=2,
    )
    assert ranked == ["B", "A"]


def test_t2_evaluator_rejects_incomplete_artifact_and_ann_matches_exact(tmp_path, monkeypatch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import evaluate_t2retrieval

    monkeypatch.setattr(evaluate_t2retrieval, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(evaluate_t2retrieval, "DATASET_DIR", tmp_path)

    def artifact(name, vectors, mapping):
        vectors = np.asarray(vectors, dtype="float32")
        mmap_path = tmp_path / f"{name}.f32.mmap"
        mmap = np.memmap(mmap_path, dtype="float32", mode="w+", shape=vectors.shape)
        mmap[:] = vectors
        mmap.flush()
        vectors_bytes = mmap_path.stat().st_size
        vectors_sha256 = hashlib.sha256(mmap_path.read_bytes()).hexdigest()
        mapping_text = "\n".join(json.dumps(row) for row in mapping) + "\n"
        mapping_bytes = len(mapping_text.encode("utf-8"))
        mapping_sha256 = hashlib.sha256(mapping_text.encode("utf-8")).hexdigest()
        exclusions_path = tmp_path / f"{name}.exclusions.jsonl"
        exclusions_path.write_text("", encoding="utf-8")
        contract = {
            "schema_version": "t2-production-rag-v2", "complete": True,
            "algorithm_version": "production-adapter-v2.2",
            "input_fingerprint": "fp", "profile_hash": "profile", "model": "m",
            "source_file": f"{name}.parquet", "source_file_sha256": "source-sha",
            "source_total_rows": len({str(row["doc_id"]) for row in mapping}),
            "source_start_row": 0,
            "source_end_row": len({str(row["doc_id"]) for row in mapping}),
            "source_selection": "all" if name == "corpus" else "prefix",
            "backend": "b", "dimension": vectors.shape[1], "shape": list(vectors.shape),
            "dtype": "float32", "vector_rows": vectors.shape[0], "document_rows": len({
                str(row["doc_id"]) for row in mapping
            }),
            "mapping_file": f"{name}.chunks.jsonl",
            "mapping_bytes": mapping_bytes,
            "mapping_sha256": mapping_sha256,
            "excluded_documents": 0,
            "exclusions_file": exclusions_path.name,
            "exclusions_bytes": 0,
            "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
            "vectors_bytes": vectors_bytes,
            "vectors_sha256": vectors_sha256,
        }
        (tmp_path / f"{name}.metadata.json").write_text(json.dumps(contract), encoding="utf-8")
        (tmp_path / f"{name}.embedding.progress.json").write_text(
            json.dumps({**contract, "next_vector_row": vectors.shape[0], "complete": True}),
            encoding="utf-8",
        )
        (tmp_path / f"{name}.prepare.progress.json").write_text(
            json.dumps({
                "complete": True, "schema_version": contract["schema_version"],
                "algorithm_version": contract["algorithm_version"],
                "input_fingerprint": contract["input_fingerprint"],
                "profile_hash": contract["profile_hash"], "artifact_rows": vectors.shape[0],
                "source_file": contract["source_file"],
                "source_file_sha256": contract["source_file_sha256"],
                "source_total_rows": contract["source_total_rows"],
                "source_start_row": contract["source_start_row"],
                "source_end_row": contract["source_end_row"],
                "source_selection": contract["source_selection"],
                "mapping_bytes": mapping_bytes, "mapping_sha256": mapping_sha256,
                "excluded_documents": 0, "exclusions_file": exclusions_path.name,
                "exclusions_bytes": 0, "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
            }), encoding="utf-8"
        )
        (tmp_path / f"{name}.chunks.jsonl").write_text(
            mapping_text, encoding="utf-8"
        )

    artifact(
        "corpus",
        [[1, 0], [0.8, 0.2], [0, 1]],
        [
            {"chunk_id": "a1", "parent_id": "pa", "doc_id": "A"},
            {"chunk_id": "a2", "parent_id": "pa", "doc_id": "A"},
            {"chunk_id": "b1", "parent_id": "pb", "doc_id": "B"},
        ],
    )
    artifact("queries", [[1, 0]], [{"chunk_id": "q", "doc_id": "q1"}])
    pq.write_table(
        pa.table({"query-id": ["q1"], "corpus-id": ["A"], "score": [1]}),
        tmp_path / "qrels.parquet",
    )
    common = dict(top_k=2, candidate_top_k=10, query_batch=1, corpus_batch=2,
                  aggregation="max", top_n=3, enforce_scope=False, expected_query_count=None)
    exact = evaluate_t2retrieval.evaluate(**common, engine="exact")
    ann = evaluate_t2retrieval.evaluate(**common, engine="faiss-hnsw")
    assert exact["Recall@2"] == ann["Recall@2"] == 1.0
    assert exact["qrels_coverage"] == 1.0

    pq.write_table(
        pa.table({"query-id": ["q1"], "corpus-id": ["B"], "score": [1]}),
        tmp_path / "qrels.parquet",
    )
    below_recall_cutoff = evaluate_t2retrieval.evaluate(
        **{**common, "top_k": 1}, engine="exact"
    )
    assert below_recall_cutoff["Recall@1"] == 0.0
    assert below_recall_cutoff["MRR@10"] == 0.5
    assert below_recall_cutoff["nDCG@10"] > 0.0

    exclusions_path = tmp_path / "corpus.exclusions.jsonl"
    exclusions_path.write_text("corrupt", encoding="utf-8")
    try:
        evaluate_t2retrieval._load_artifact("corpus")
    except ValueError as exc:
        assert "exclusions digest/byte contract" in str(exc)
    else:
        raise AssertionError("corrupt exclusions must not be evaluated")
    exclusions_path.write_text("", encoding="utf-8")

    progress_path = tmp_path / "corpus.embedding.progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["complete"] = False
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    try:
        evaluate_t2retrieval._load_artifact("corpus")
    except ValueError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("incomplete vector progress must not be evaluated")


def test_t2_embedding_resume_rejects_missing_mmap_and_short_committed_jsonl(tmp_path, monkeypatch):
    import pytest
    from dataset import embed_t2retrieval

    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path)
    manifest_path = tmp_path / "corpus.chunks.jsonl"
    mapping_text = json.dumps({"embedding_text": "content"}) + "\n"
    manifest_path.write_text(mapping_text, encoding="utf-8")
    exclusions_path = tmp_path / "corpus.exclusions.jsonl"
    exclusions_path.write_text("", encoding="utf-8")
    prepared = {
        "algorithm_version": "production-adapter-v2.2",
        "input_fingerprint": "fp",
        "profile_hash": "profile",
        "source_file": "corpus.parquet",
        "source_file_sha256": "source-sha",
        "source_total_rows": 1,
        "source_start_row": 0,
        "source_end_row": 1,
        "source_selection": "all",
        "total_documents": 1,
        "artifact_rows": 1,
        "mapping_bytes": len(mapping_text.encode("utf-8")),
        "mapping_sha256": hashlib.sha256(mapping_text.encode("utf-8")).hexdigest(),
        "excluded_documents": 0,
        "exclusions_file": exclusions_path.name,
        "exclusions_bytes": 0,
        "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
    }
    model = FakeEmbedding()
    embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)
    mmap_path = tmp_path / "corpus.f32.mmap"
    with mmap_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))
    with pytest.raises(ValueError, match="digest does not match committed contract"):
        embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)

    mmap_path.unlink()
    with pytest.raises(ValueError, match="mmap is missing"):
        embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)

    short_path = tmp_path / "short.jsonl"
    short_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="shorter than committed progress"):
        embed_t2retrieval._require_committed_prefix(short_path, 10, "fixture")


def test_t2_prepare_recovers_uncommitted_suffix_and_never_recertifies_complete_mapping(
    tmp_path, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter
    from dataset import embed_t2retrieval

    monkeypatch.setattr(embed_t2retrieval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path / "out")
    pq.write_table(
        pa.table(
            {
                "_id": ["one", "two", "bad"],
                "text": ["第一篇有效正文。", "第二篇有效正文。", " </pre>"],
                "title": ["", "", ""],
            }
        ),
        tmp_path / "corpus.parquet",
    )
    base = ProductionRagAdapter()

    class FailSecondDocument:
        chunk_config = base.chunk_config
        text_builder = base.text_builder

        def __init__(self):
            self.calls = 0

        def child_inputs(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("unexpected adapter failure")
            return base.child_inputs(**kwargs)

    with pytest.raises(ValueError, match="unexpected adapter failure"):
        embed_t2retrieval._prepare_manifest("corpus", FailSecondDocument(), None)

    manifest_path = tmp_path / "out/corpus.chunks.jsonl"
    exclusions_path = tmp_path / "out/corpus.exclusions.jsonl"
    with manifest_path.open("ab") as handle:
        handle.write(b'{"uncommitted":')
    with exclusions_path.open("ab") as handle:
        handle.write(b'{"uncommitted":')

    _, progress = embed_t2retrieval._prepare_manifest("corpus", base, None)
    assert progress["complete"] is True
    assert progress["excluded_documents"] == 1
    assert json.loads(exclusions_path.read_text(encoding="utf-8"))["doc_id"] == "bad"

    original = manifest_path.read_bytes()
    corrupted = original.replace(b"Content", b"content", 1)
    assert len(corrupted) == len(original) and corrupted != original
    manifest_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="digest does not match committed contract"):
        embed_t2retrieval._prepare_manifest("corpus", base, None)
