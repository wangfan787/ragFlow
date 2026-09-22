"""解析边界：内存输入、Markdown 特殊结构、编码错误和来源独立性。"""
from pathlib import Path

from backend.src.parsing.parser_factory import build_parser


def test_text_and_html_parsers_accept_in_memory_sources(tmp_path: Path):
    text_blocks = build_parser("text").parse(
        "text-doc", {"file_type": "text", "text": "first\n\nsecond", "doc_name": "x.txt"}
    )
    assert [row.page_content for row in text_blocks] == ["first", "second"]
    assert text_blocks[1].metadata["source_span"]["accuracy"] == "exact"

    html_blocks = build_parser("html").parse(
        "html-doc",
        {
            "file_type": "html",
            "text": "<h1>Title</h1><script>bad()</script><p>Hello &amp; world</p>",
            "doc_name": "x.html",
        },
    )
    assert [row.metadata["block_type"] for row in html_blocks] == ["heading", "paragraph"]
    assert "bad" not in " ".join(row.page_content for row in html_blocks)
    assert html_blocks[1].metadata["source_span"]["accuracy"] == "line_only"
    loose_blocks = build_parser("html").parse(
        "loose-html",
        {"file_type": "html", "text": "before<div><p>inside</p>after</div>", "doc_name": "x"},
    )
    assert [row.page_content for row in loose_blocks] == ["before", "inside", "after"]
    repeated = build_parser("html").parse(
        "repeat", {"file_type": "html", "text": "<p>same</p>\n<p>same</p>", "doc_name": "x"}
    )
    # unstructured 返回规范化后的元素文本，回定位坐标指向文本本身而非外层
    # <p> 标签；不变量是重复片段必须定位到不同位置（游标单调前进）。
    assert repeated[0].metadata["source_span"]["start_char"] > 0
    assert repeated[1].metadata["source_span"]["start_char"] > repeated[0].metadata["source_span"]["end_char"]


def test_markdown_preserves_indented_code_and_html_blocks(tmp_path: Path):
    source = tmp_path / "mixed.md"
    source.write_text(
        "before\n\n    secret_code()\n\n<div>secret html</div>\n\nafter",
        encoding="utf-8",
    )
    blocks = build_parser("md").parse(
        "mixed", {"file_type": "md", "file_path": str(source), "doc_name": source.name}
    )
    content = "\n".join(row.page_content for row in blocks)
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
    content = "\n".join(row.page_content for row in blocks)
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


def test_parser_metadata_is_independent_between_blocks():
    from backend.src.parsing.parser_factory import build_parser

    blocks = build_parser("text").parse("d", {"file_type": "text", "text": "first\n\nsecond"})
    blocks[0].metadata["source_span"]["start_line"] = 99
    assert blocks[1].metadata["source_span"]["start_line"] == 3
