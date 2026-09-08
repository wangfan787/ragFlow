"""解析器回归：真实样例返回 Document，来源字段可读，失败让 pytest 报错。"""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.src.parsing.parser_factory import build_parser

DATA = Path(__file__).parent


@pytest.mark.parametrize("kind,name", [("md", "并发编程-锁.md"), ("pdf", "简历.pdf")])
def test_sample_parser_returns_documents_with_provenance(kind, name):
    path = DATA / name
    blocks = build_parser(kind).parse(
        "sample", {"file_type": kind, "file_path": str(path), "doc_name": name},
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
