"""真实 Markdown/PDF 样例的父子切片守恒测试。"""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.src.parsing.parser_factory import build_parser
from backend.src.chunking import ChunkConfig, BlockChunker
from backend.src.chunking.token_counter import SimpleTokenCounter

DATA = Path(__file__).parents[1] / "parsing" / "data"


@pytest.mark.parametrize("kind,name", [("md", "并发编程-锁.md"), ("pdf", "简历.pdf")])
def test_sample_chunking_preserves_parent_text_and_links(kind, name):
    blocks = build_parser(kind).parse(
        "sample", {"file_type": kind, "file_path": str(DATA / name), "doc_name": name},
    )
    assert blocks and all(isinstance(block, Document) for block in blocks)
    assert all(block.page_content.strip() for block in blocks)
    assert all(block.metadata["doc_id"] == "sample" for block in blocks)
    assert all(block.metadata["doc_name"] == name for block in blocks)
    for block in blocks:
        span = block.metadata["source_span"]
        assert span and span.get("accuracy", "unavailable") in {"exact", "line_only", "unavailable"}
        if kind == "pdf":
            assert block.metadata["page_no"] > 0
        else:
            assert span["start_line"] >= 1
            assert span["end_line"] >= span["start_line"]
    config = ChunkConfig()
    chunks = BlockChunker().chunk(blocks, config)
    assert chunks and all(isinstance(chunk, Document) for chunk in chunks)
    parents = [chunk for chunk in chunks if chunk.metadata["chunk_role"] == "parent"]
    children = [chunk for chunk in chunks if chunk.metadata["chunk_role"] == "child"]
    assert parents and children
    assert all(chunk.metadata["preserve_structure"] or chunk.metadata["token_count"] <= config.child_max_tokens for chunk in children)
    for parent in parents:
        assert parent.metadata["preserve_structure"] or parent.metadata["token_count"] <= config.parent_max_tokens
        family = [child for child in children if child.metadata["parent_id"] == parent.metadata["chunk_id"]]
        assert "".join(child.page_content for child in family) == parent.page_content
        assert parent.metadata["child_ids"] == [child.metadata["chunk_id"] for child in family]
        assert parent.metadata["retrieval_eligible"] is False
        for child in family:
            metadata = child.metadata
            assert metadata["retrieval_eligible"] is True
            assert parent.page_content[metadata["parent_char_start"]:metadata["parent_char_end"]] == child.page_content


def _document(content="", **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


def _block(text: str, block_type: str = "paragraph") -> Document:
    return _document(
        text, doc_id="doc", block_type=block_type, section_path=["section"], order=1,
        source_span={"start_line": 1, "end_line": 1, "start_char": 0,
                     "end_char": len(text), "accuracy": "exact"},
    )


def test_strict_chunking_conserves_unpunctuated_code_and_table():
    counter = SimpleTokenCounter()
    config = ChunkConfig(
        parent_target_tokens=40,
        parent_max_tokens=50,
        child_target_tokens=12,
        child_max_tokens=16,
        embedding_input_budget=64,
        preserve_code_block=False,
        preserve_table_block=False,
    )
    for block_type, text in (
        ("paragraph", "无标点中文" * 100),
        ("code", "identifier_without_spaces=" + "x" * 500),
        ("table", "|cell|" * 300),
    ):
        chunks = BlockChunker().chunk([_block(text, block_type)], config)
        parents = [row for row in chunks if row.metadata["chunk_role"] == "parent"]
        children = [row for row in chunks if row.metadata["chunk_role"] == "child"]
        assert parents and children
        assert all(counter.count(row.page_content) <= config.parent_max_tokens for row in parents)
        assert all(counter.count(row.page_content) <= config.child_max_tokens for row in children)
        assert all(not row.metadata["retrieval_eligible"] for row in parents)
        assert all(row.metadata["retrieval_eligible"] for row in children)
        for parent in parents:
            family = [row for row in children if row.metadata["parent_id"] == parent.metadata["chunk_id"]]
            assert "".join(row.page_content for row in family) == parent.page_content


def test_child_serialization_preserves_boundary_whitespace_and_exact_span():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    config = ChunkConfig(
        parent_target_tokens=40,
        parent_max_tokens=50,
        child_target_tokens=3,
        child_max_tokens=4,
        embedding_input_budget=50,
    )
    chunks = BlockChunker().chunk([_block(text)], config)
    parent = next(row for row in chunks if row.metadata["chunk_role"] == "parent")
    children = [row for row in chunks if row.metadata["parent_id"] == parent.metadata["chunk_id"]]
    assert "".join(row.page_content for row in children) == parent.page_content == text
    for child in children:
        span = child.metadata["source_span"]
        assert text[span["start_char"] : span["end_char"]] == child.page_content
