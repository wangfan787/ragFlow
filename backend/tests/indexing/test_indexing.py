"""索引与 ES 边界：正文输入、父子存储、非法向量、过滤和重建预检。"""
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.chunking import ChunkConfig, BlockChunker
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore


class FakeEmbedding:
    model = "fake-v1"
    dimensions = 2

    def __init__(self):
        self.inputs: list[str] = []

    def embed_documents(self, texts):
        self.inputs.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeStore:
    def __init__(self, records=None):
        self.records = records or []
        self.upserted = []

    def delete_by_doc_id(self, doc_id):
        return None

    def delete_stale_by_doc_id(self, doc_id, keep_ids):
        return None

    def upsert(self, records, vectors):
        self.upserted = records
        self.vectors = vectors

    def query_by_ids(self, ids):
        return [record for record in self.records if record.metadata["chunk_id"] in ids]

    def vector_search(self, query_vector, top_k, filters=None):
        return []

    def keyword_search(self, query, top_k, filters=None):
        return []


def _document(content="", **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


def _block(text: str, block_type: str = "paragraph") -> Document:
    return _document(
        text, doc_id="doc", block_type=block_type, section_path=["section"], order=1,
        source_span={"start_line": 1, "end_line": 1, "start_char": 0,
                     "end_char": len(text), "accuracy": "exact"},
    )


def test_indexer_embeds_only_children_and_keeps_parent_context_record():
    chunks = BlockChunker().chunk([_block("正文内容。" * 80)], ChunkConfig())
    for chunk in chunks:
        chunk.metadata["doc_name"] = "doc.md"
        chunk.metadata["question_kwd"] = ["问题"]
    embedding = FakeEmbedding()
    store = FakeStore()
    count = EmbeddingIndexer(embedding_model=embedding, store=store).index("doc", chunks, "doc.md")
    children = [row for row in chunks if row.metadata["retrieval_eligible"]]
    assert count == len(chunks)
    assert len(embedding.inputs) == len(children)
    assert embedding.inputs == [child.page_content for child in children]
    assert all(record.metadata["chunk_id"] not in store.vectors for record in store.upserted if not record.metadata["retrieval_eligible"])
    assert all(store.vectors[record.metadata["chunk_id"]] for record in store.upserted if record.metadata["retrieval_eligible"])


def test_indexer_rejects_oversized_body_before_model_call():
    chunks = BlockChunker().chunk([_block("content " * 60)], ChunkConfig())
    for chunk in chunks:
        chunk.metadata["embedding_input_budget"] = 1
    embedding = FakeEmbedding()
    store = FakeStore()
    with pytest.raises(ValueError, match="重新切片"):
        EmbeddingIndexer(embedding_model=embedding, store=store).index("doc", chunks, "doc")
    assert embedding.inputs == []
    assert store.upserted == []


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[0.0, 0.0]], [[float("nan"), 1.0]]])
def test_indexer_does_not_publish_invalid_model_vectors(vectors):
    model = FakeEmbedding()
    model.embed_documents = lambda texts: vectors
    store = FakeStore()
    chunks = [_document("body", chunk_id="child", retrieval_eligible=True, embedding_input_budget=192)]
    with pytest.raises(ValueError):
        EmbeddingIndexer(embedding_model=model, store=store).index("doc", chunks)
    assert store.upserted == []


@pytest.mark.parametrize("vector", [[], [1.0], [0.0, 0.0], [float("nan"), 1.0]])
def test_retriever_rejects_invalid_vector_before_search(vector):
    from backend.src.retrieval.embedding_retriever import EmbeddingRetriever

    model = FakeEmbedding()
    model.embed_query = lambda text: vector
    store = FakeStore()
    store.vector_search = lambda *args, **kwargs: pytest.fail("invalid vector reached ES")
    with pytest.raises(ValueError):
        EmbeddingRetriever(embedding_model=model, store=store).retrieve("question")


def test_production_index_and_retrieval_build_langchain_only_when_called(monkeypatch):
    from backend.src.indexing import embedding_indexer
    from backend.src.retrieval import embedding_retriever

    calls = []
    model = FakeEmbedding()

    def factory():
        calls.append("build")
        return model

    monkeypatch.setattr(embedding_indexer, "build_embeddings", factory)
    monkeypatch.setattr(embedding_retriever, "build_embeddings", factory)
    indexer = EmbeddingIndexer(store=FakeStore())
    retriever = embedding_retriever.EmbeddingRetriever(store=FakeStore())
    assert calls == []
    indexer.index("doc", [_document(" body ", chunk_id="c", retrieval_eligible=True, embedding_input_budget=192)])
    retriever.retrieve("question")
    assert calls == ["build", "build"]
    assert model.inputs == [" body ", "question"]


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
