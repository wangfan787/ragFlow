"""P0-A：Markdown 本地图片引用提取与 image block 交错的纯文本契约。

验收对应 docs/plan/算法层QA.md §7.2 P0-A：
- 只接受相对路径本地 PNG/JPEG/WebP；
- URL / 绝对路径 / `..` 穿越 / 不支持类型 / 缺文件全部显式 skip 并带原因；
- 围栏代码块中的图片语法不算引用；
- source_span 精度 exact，可与文本块坐标交错。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from backend.src.parsing.markdown_images import extract_image_refs, interleave_image_blocks


def _write_image(directory: Path, name: str, content: bytes = b"fake-bytes") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _text_block(block_type: str, start: int, end: int, section: list[str] | None = None) -> Document:
    return Document(
        page_content=f"{block_type}-{start}",
        metadata={
            "block_type": block_type,
            "section_path": list(section or []),
            "order": 0,
            "source_span": {
                "start_line": 1, "end_line": 1,
                "start_char": start, "end_char": end,
                "page_no": None, "accuracy": "line_only",
            },
        },
    )


def _image_document(start: int, end: int) -> Document:
    return Document(
        page_content="[图片] 测试",
        metadata={
            "block_type": "image",
            "section_path": [],
            "order": 0,
            "source_span": {
                "start_line": 1, "end_line": 1,
                "start_char": start, "end_char": end,
                "page_no": None, "accuracy": "exact",
            },
        },
    )


def test_extracts_relative_image_with_exact_span(tmp_path: Path) -> None:
    _write_image(tmp_path, "arch.png")
    text = "# 标题\n\n前文\n\n![架构图](arch.png)\n\n后文\n"
    refs = extract_image_refs(text, tmp_path)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.alt == "架构图"
    assert ref.skip_reason is None
    assert ref.mime_type == "image/png"
    assert ref.resolved_path == (tmp_path / "arch.png").resolve()
    # span 必须逐字对应原文中的图片语法
    assert text[ref.start_char:ref.end_char] == "![架构图](arch.png)"
    assert ref.start_line == 5


def test_skip_rules_are_explicit(tmp_path: Path) -> None:
    _write_image(tmp_path, "ok.png")
    text = "\n".join(
        [
            "![外链](https://example.com/a.png)",
            "![数据](data:image/png;base64,xxxx)",
            "![绝对](/etc/passwd.png)",
            "![穿越](../outside.png)",
            "![类型](diagram.svg)",
            "![缺失](missing.png)",
            "![锚点](#section)",
            "![正常](ok.png)",
        ]
    )
    refs = extract_image_refs(text, tmp_path)
    reasons = [ref.skip_reason for ref in refs if ref.skip_reason is not None]
    assert reasons == [
        "external_url", "external_url", "absolute_path",
        "path_traversal", "unsupported_type", "file_not_found", "anchor",
    ]
    candidates = [ref for ref in refs if ref.resolved_path is not None]
    assert [ref.raw_path for ref in candidates] == ["ok.png"]


def test_fenced_code_images_are_ignored(tmp_path: Path) -> None:
    _write_image(tmp_path, "ok.png")
    text = "```md\n![围栏内](ok.png)\n```\n\n![围栏外](ok.png)\n"
    refs = extract_image_refs(text, tmp_path)
    assert [ref.alt for ref in refs] == ["围栏外"]


def test_supported_extensions_map_mime_types(tmp_path: Path) -> None:
    for name, mime in [
        ("a.png", "image/png"), ("b.jpg", "image/jpeg"),
        ("c.jpeg", "image/jpeg"), ("d.webp", "image/webp"),
    ]:
        _write_image(tmp_path, name)
    text = "\n".join(f"![x]({name})" for name in ["a.png", "b.jpg", "c.jpeg", "d.webp"])
    refs = extract_image_refs(text, tmp_path)
    assert [ref.mime_type for ref in refs] == ["image/png", "image/jpeg", "image/jpeg", "image/webp"]


def test_interleave_orders_renumbers_and_inherits_section() -> None:
    text_blocks = [
        _text_block("heading", 0, 10, section=["检索"]),
        _text_block("paragraph", 20, 40, section=["检索"]),
        _text_block("paragraph", 60, 80, section=["检索"]),
    ]
    images = [_image_document(45, 55), _image_document(100, 110)]
    merged = interleave_image_blocks(text_blocks, images)
    assert [block.metadata["block_type"] for block in merged] == [
        "heading", "paragraph", "image", "paragraph", "image",
    ]
    assert [block.metadata["order"] for block in merged] == [1, 2, 3, 4, 5]
    assert images[0].metadata["section_path"] == ["检索"]


def test_interleave_drops_alt_text_paragraph_inside_image_span() -> None:
    text_blocks = [
        _text_block("paragraph", 0, 5),
        _text_block("paragraph", 12, 25),  # 落在图片语法区间内（alt 文本被引擎单独输出）
        _text_block("paragraph", 60, 70),
    ]
    merged = interleave_image_blocks(text_blocks, [_image_document(10, 30)])
    assert [block.metadata["block_type"] for block in merged] == ["paragraph", "image", "paragraph"]
    assert [block.metadata["order"] for block in merged] == [1, 2, 3]
