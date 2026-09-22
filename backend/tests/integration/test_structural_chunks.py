"""验证结构从 Markdown/HTML 解析、入库到 QA 证据保持完整。"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.apps.services.common_service import ServiceError
from backend.src.apps.services.qa_service import QAService
from backend.src.chunking import BlockChunker, ChunkConfig
from backend.src.chunking.token_counter import count_tokens
from backend.src.config.qa_config import QAConfig
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.indexing.embedding_text_builder import EmbeddingTextBuilder
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.parsing.parser_factory import build_parser
from backend.src.retrieval.hybrid_router import HybridRouter


def parse(text, kind="md"):
    return build_parser(kind).parse("doc", {"file_type": kind, "text": text, "doc_name": "example"})


@pytest.mark.parametrize("structure,kind", [
    ("```python\nif True:\n    foo()\n```\n", "code"),
    ("  ~~~python\n  if True:\n      foo()\n  ~~~\n", "code"),
    ("      if True:\n          foo()\n", "code"),
    ("> ```python\n> foo()\n> ```\n", "code"),
    ("Name | Value\n--- | ---\na | b\n", "table"),
    ("| Name | Value |\n| --- | --- |\n| a | b |\n", "table"),
    ("```md\n| a | b |\n| - | - |\n| c | d |\n```\n", "code"),
    ("```python\nfoo()\n", "code"),
])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_markdown_preserves_structures_and_exact_source_spans(structure, kind, newline):
    text = ("---\nname: sample\n---\n# Section\n\n- item\n\n" + structure).replace("\n", newline)
    blocks = parse(text)
    matches = [b for b in blocks if b.metadata["block_type"] == kind]
    assert len(matches) == 1
    block = matches[0]
    assert block.page_content == structure.replace("\n", newline)
    assert block.metadata["section_path"] == ["Section"]
    span = block.metadata["source_span"]
    assert span["accuracy"] == "exact"
    assert text[span["start_char"]:span["end_char"]] == block.page_content
    assert not any("ragflowprotectedstructure" in b.page_content for b in blocks)


def test_html_table_keeps_cells_and_headers():
    blocks = parse("<h1>Title</h1><table><tr><th>Name</th><th>Value</th></tr>"
                   "<tr><td>alpha</td><td>42</td></tr></table>", "html")
    table = next(b for b in blocks if b.metadata["block_type"] == "table")
    assert "<table>" in table.page_content and "<tr>" in table.page_content
    assert "Name" in table.page_content and "42" in table.page_content


@pytest.mark.parametrize("kind", ["code", "table"])
def test_complete_structure_survives_index_retrieval_and_qa(kind):
    body = (
        "```python\n" + "    result = calculate(value)\n" * 140 + "```\n"
        if kind == "code" else "| Name | Value |\n| --- | --- |\n" + "| alpha | beta |\n" * 220
    )
    assert 768 < count_tokens(body) < 3072
    assert len(body.encode()) > 3072
    chunks = BlockChunker().chunk(parse("# Section\n\nbefore\n\n" + body + "\nafter"), ChunkConfig())
    family = [c for c in chunks if c.metadata["preserve_structure"]]
    assert len(family) == 2
    parent, child = family
    assert parent.page_content == child.page_content == body
    assert parent.metadata["child_ids"] == [child.metadata["chunk_id"]]
    assert child.metadata["parent_id"] == parent.metadata["chunk_id"]
    assert child.metadata["parent_char_start"] == 0
    assert child.metadata["parent_char_end"] == len(body)
    assert all(c.metadata["block_types"] == [kind] for c in family)

    inputs, stored = [], []

    def embed(texts):
        inputs.extend(texts)
        return [[1.0, 0.5] for _ in texts]

    model = SimpleNamespace(model="test", dimensions=2, embed_documents=embed)
    store = SimpleNamespace(
        upsert=lambda records, vectors: stored.extend(records),
        delete_stale_by_doc_id=lambda *args: None,
    )
    EmbeddingIndexer(embedding_model=model, store=store).index("doc", chunks)
    assert inputs.count(body) == 1
    # 模拟 ES 序列化/反序列化，验证保护标记能经过存储进入父块展开。
    records = [ElasticsearchStore._document({
        "_id": c.metadata["chunk_id"], "_source": {**c.metadata, "content": c.page_content},
    }) for c in stored]
    router = HybridRouter(store=SimpleNamespace(query_by_ids=lambda ids: [
        r for r in records if r.metadata["chunk_id"] in ids
    ]), embedding_model=model)
    rows, _ = router._expand_parent_context([
        {**child.metadata, "content": body, "score": 0.8, "vector_score": 0.8},
    ])
    expanded = Document(page_content=rows[0]["content"], metadata={
        k: v for k, v in rows[0].items() if k != "content"
    })
    service = QAService(model=SimpleNamespace())
    service.context_limit_tokens = 8192
    service.completion_reserve_tokens = 512
    service.prompt_safety_tokens = 256
    qa = QAConfig(context_top_k=1, evidence_window_tokens=32)
    evidence, _ = service._budgeted_evidence("解释内容", [expanded], qa)
    assert [e.page_content for e in evidence] == [body]
    service.context_limit_tokens = 900
    with pytest.raises(ServiceError) as error:
        service._budgeted_evidence("解释内容", [expanded], qa)
    assert error.value.code == "QA_CONTEXT_BUDGET_EXHAUSTED"
    assert expanded.page_content == body


def test_structure_over_model_budget_is_rejected_before_embedding_or_storage():
    body = "```python\n" + "x = calculate(value)\n" * 700 + "```\n"
    assert count_tokens(body) > 3072
    chunks = BlockChunker().chunk(parse(body), ChunkConfig())
    assert len(chunks) == 2 and chunks[1].page_content == body
    model = SimpleNamespace(embed_documents=lambda texts: pytest.fail("must not call provider"))
    store = SimpleNamespace(upsert=lambda *args: pytest.fail("must not publish"))
    with pytest.raises(ValueError, match="模型嵌入输入上限"):
        EmbeddingIndexer(embedding_model=model, store=store).index("doc", chunks)
    assert chunks[1].page_content == body


def test_plain_text_still_obeys_chunk_budget_and_configurable_embedding_budget():
    config = ChunkConfig(child_target_tokens=256, child_max_tokens=384, embedding_input_budget=384)
    chunks = BlockChunker().chunk(parse("word " * 1000, "txt"), config)
    children = [c for c in chunks if c.metadata["retrieval_eligible"]]
    assert any(c.metadata["token_count"] > 192 for c in children)
    assert all(c.metadata["token_count"] <= 384 for c in children)
    builder = EmbeddingTextBuilder()
    assert all(builder.build(c).text == c.page_content for c in children)
    children[0].metadata["embedding_input_budget"] = 1
    with pytest.raises(ValueError, match="重新切片"):
        builder.build(children[0])
