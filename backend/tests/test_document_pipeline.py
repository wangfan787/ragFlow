"""Document 贯穿实际业务入口；只替换外部嵌入、聊天及 ES 连接。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.apps.services.ingestion_pipeline import IngestionPipeline
from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.apps.services.qa_service import QAService
from backend.src.chunking import ChunkConfig
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.retrieval.hybrid_router import HybridRouter


class MemoryStore:
    def __init__(self):
        self.records = []
        self.vectors = {}

    def upsert(self, records, vectors):
        self.records = records
        self.vectors = vectors

    def delete_stale_by_doc_id(self, doc_id, keep_ids):
        assert set(keep_ids) == {record.metadata["chunk_id"] for record in self.records}

    def query_by_ids(self, ids):
        return [record for record in self.records if record.metadata["chunk_id"] in ids]

    def vector_search(self, vector, top_k, filters=None):
        assert filters["retrieval_eligible"] is True
        return [Document(page_content=record.page_content, metadata={**record.metadata, "score": 0.8})
                for record in self.records if record.metadata["retrieval_eligible"]][:top_k]

    def keyword_search(self, query, top_k, filters=None):
        assert filters["retrieval_eligible"] is True
        return [Document(page_content=record.page_content, metadata={**record.metadata, "keyword_score": 0.6})
                for record in self.records if record.metadata["retrieval_eligible"]][:top_k]


def test_document_flow_from_parser_to_qa_preserves_body_scores_and_sources():
    embedded = []
    model = SimpleNamespace(
        model="fixture", dimensions=2,
        embed_documents=lambda texts: embedded.extend(texts) or [[1.0, 0.0] for _ in texts],
        embed_query=lambda question: [1.0, 0.0],
    )
    store = MemoryStore()
    pipeline = IngestionPipeline(embedding_indexer=EmbeddingIndexer(model, store))
    config = {"file_type": "md", "doc_name": "notes.md", "text": "# 锁\n\n互斥锁保护共享资源。\n\n自旋锁会忙等。"}
    result = pipeline.run("doc", config, ChunkConfig())
    assert result["indexed_count"] == len(store.records)
    assert all(isinstance(record, Document) for record in store.records)
    children = [record for record in store.records if record.metadata["retrieval_eligible"]]
    assert embedded == [record.page_content for record in children]
    assert set(store.vectors) == {record.metadata["chunk_id"] for record in children}
    assert all(not {"content", "text", "meta"}.intersection(record.metadata) for record in store.records)
    assert all(record.metadata["source_span"]["start_line"] >= 1 for record in store.records)

    router = HybridRouter(store=store, embedding_model=model)
    parents = router.retrieve("互斥锁的作用")
    assert parents and all(isinstance(parent, Document) for parent in parents)
    assert all(parent.metadata["score"] == pytest.approx(0.75) for parent in parents)
    assert sum(len(parent.metadata["matched_children"]) for parent in parents) == len(children)
    messages = []
    qa = QAService(model=SimpleNamespace(invoke=lambda value: messages.extend(value) or SimpleNamespace(content="互斥锁保护共享资源。[1]")))
    qa.retriever = router
    answer = qa.query("互斥锁的作用")
    citation = answer["citations"][0]
    assert "互斥锁保护共享资源" in messages[1]["content"]
    assert citation["doc_id"] == "doc"
    assert citation["source_span"]["start_line"] >= 1
    assert citation["matched_children"]
    assert answer["trace"]["citation_avg_score"] == pytest.approx(0.75)


def test_es_boundary_roundtrip_keeps_metadata_and_parent_has_no_vector():
    requests = []
    client = SimpleNamespace(
        indices=SimpleNamespace(exists=lambda **kwargs: True),
        bulk=lambda **kwargs: requests.append(kwargs) or {"errors": False},
    )
    store = ElasticsearchStore(client=client, index_name="fixture")
    store._ensure_index = lambda dimension: f"q_{dimension}_vec"
    parent = Document(page_content="parent body", metadata={
        "chunk_id": "p", "doc_id": "d", "retrieval_eligible": False,
        "source_span": {"start_line": 3, "end_line": 6}, "custom_source": {"asset_id": "image-1"},
    })
    child = Document(page_content="body", metadata={
        "chunk_id": "c", "doc_id": "d", "retrieval_eligible": True, "parent_id": "p",
        "parent_char_start": 7, "parent_char_end": 11,
    })
    store.upsert([parent, child], {"c": [1.0, 0.0]})
    operations = requests[0]["operations"]
    assert "q_2_vec" not in operations[1]
    assert operations[3]["q_2_vec"] == [1.0, 0.0]
    client.mget = lambda **kwargs: {"docs": [
        {"_id": "p", "_source": operations[1], "found": True},
        {"_id": "c", "_source": operations[3], "found": True},
        {"_id": "missing", "found": False},
    ]}
    loaded = store.query_by_ids(["p", "c", "missing"])
    assert loaded[0].page_content == parent.page_content
    assert loaded[0].metadata == parent.metadata
    assert loaded[1].metadata["parent_char_start"] == 7
    assert "q_2_vec" not in loaded[1].metadata


def test_window_keeps_input_unchanged_and_uses_absolute_parent_coordinates():
    text = "前文" * 1000 + "命中证据" + "后文" * 1000
    start = text.index("命中证据")
    original = Document(page_content=text, metadata={
        "chunk_id": "p", "doc_id": "d", "chunk_role": "parent", "score": 0.8,
        "matched_children": [{"chunk_id": "c", "snippet": "命中证据",
                              "parent_char_start": start, "parent_char_end": start + 4}],
        "source_span": {"start_line": 1, "end_line": 20},
    })
    before = deepcopy(original.model_dump())
    builder = ContextWindowBuilder()
    first = builder.anchored(original, 100)
    second = builder.anchored(first, 40)
    assert original.model_dump() == before
    assert "命中证据" in second.page_content
    span = second.metadata["prompt_span"]
    assert text[span["parent_char_start"]:span["parent_char_end"]] == second.page_content


def test_parser_metadata_is_independent_between_blocks():
    from backend.src.parsing.parser_factory import build_parser

    blocks = build_parser("text").parse("d", {"file_type": "text", "text": "first\n\nsecond"})
    blocks[0].metadata["source_span"]["start_line"] = 99
    assert blocks[1].metadata["source_span"]["start_line"] == 3
