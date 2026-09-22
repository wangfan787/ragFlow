"""真实 Markdown/PDF 样例的父子切片守恒测试。"""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.src.parsing.parser_factory import build_parser
from backend.src.chunking import ChunkConfig, BlockChunker

DATA = Path(__file__).parents[1] / "parsing" / "data"


@pytest.mark.parametrize("kind,name", [("md", "并发编程-锁.md"), ("pdf", "简历.pdf")])
def test_sample_chunking_preserves_parent_text_and_links(kind, name):
    blocks = build_parser(kind).parse(
        "sample", {"file_type": kind, "file_path": str(DATA / name), "doc_name": name},
    )
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
