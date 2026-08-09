#!/usr/bin/env python3
"""对比新旧 Markdown 解析器的测试脚本。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.parsing.markdown_parser import MarkdownParser
from backend.src.parsing.markdown_parser_it import MarkdownParserIt

# 测试 Markdown 文件
TEST_MD = PROJECT_ROOT / "backend" / "tests" / "parsing_test" / "sample_walkthrough.md"
DOC_ID = "test_doc"

def test_parser(parser_class, name: str):
    """测试指定的解析器。"""
    print(f"\n{'=' * 80}")
    print(f"测试解析器: {name}")
    print(f"{'=' * 80}")

    parser = parser_class()
    blocks = parser.parse(
        doc_id=DOC_ID,
        parse_config={"file_type": "md", "file_path": str(TEST_MD), "doc_name": TEST_MD.name}
    )

    print(f"\n解析出 {len(blocks)} 个 block：\n")

    # 按类型统计
    type_counts = {}
    for block in blocks:
        block_type = block["block_type"]
        type_counts[block_type] = type_counts.get(block_type, 0) + 1

    print("类型分布:")
    for block_type, count in sorted(type_counts.items()):
        icon = {
            "frontmatter": "📋", "heading": "🏷️ ", "paragraph": "📝", "table": "📊",
            "list": "📑", "code": "💻", "blockquote": "💬", "hr": "➖",
        }.get(block_type, "  ")
        print(f"  {icon} {block_type:<12} {count:>3} 个")

    # 显示前 10 个 block
    print(f"\n前 10 个 block 详情：")
    for i, block in enumerate(blocks[:10], 1):
        block_type = block["block_type"]
        icon = {
            "frontmatter": "📋", "heading": "🏷️ ", "paragraph": "📝", "table": "📊",
            "list": "📑", "code": "💻", "blockquote": "💬", "hr": "➖",
        }.get(block_type, "  ")
        section = " > ".join(block.get("section_path", [])) or "（无章节）"
        text_preview = block["text"][:50].replace("\n", "⏎")
        if len(block["text"]) > 50:
            text_preview += "…"

        print(f"  {icon} Block #{i:2d} [{block_type:<10}]")
        print(f"             章节: {section}")
        print(f"             行号: L{block['source_span']['start_line']}-L{block['source_span']['end_line']}")
        print(f"             内容: {text_preview}")
        print()

    return blocks

if __name__ == "__main__":
    # 测试旧解析器
    old_blocks = test_parser(MarkdownParser, "旧解析器 (markdown_parser.py - 手写正则)")

    # 测试新解析器
    new_blocks = test_parser(MarkdownParserIt, "新解析器 (markdown_parser_it.py - markdown-it-py)")

    # 对比结果
    print(f"\n{'=' * 80}")
    print("对比结果")
    print(f"{'=' * 80}")
    print(f"旧解析器 block 数: {len(old_blocks)}")
    print(f"新解析器 block 数: {len(new_blocks)}")

    # 找出差异
    if len(old_blocks) != len(new_blocks):
        print(f"⚠️  block 数量不同！差异: {len(new_blocks) - len(old_blocks)}")
    else:
        print("✅ block 数量相同")

    # 逐个对比
    print(f"\n逐个 block 对比：")
    for i, (old, new) in enumerate(zip(old_blocks, new_blocks), 1):
        old_type = old["block_type"]
        new_type = new["block_type"]
        if old_type != new_type:
            print(f"  ⚠️  Block #{i}: 类型不同 {old_type} → {new_type}")
        else:
            print(f"  ✅ Block #{i}: 类型相同 [{old_type}]")
