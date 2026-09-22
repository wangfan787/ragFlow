"""Document 贯穿实际业务入口；只替换外部嵌入、聊天及 ES 连接。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore


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
