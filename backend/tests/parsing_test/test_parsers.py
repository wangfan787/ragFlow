#!/usr/bin/env python3
"""
Parsing 模块测试脚本
测试 Markdown 和 PDF 解析器的解析效果
"""

from __future__ import annotations
import sys
from pathlib import Path
from pprint import pprint

# 添加项目路径到 Python 模块搜索路径
project_root = Path(__file__).resolve().parents[3]  # 回到项目根目录
sys.path.insert(0, str(project_root))

from backend.src.parsing.parser_factory import build_parser
from backend.src.parsing.models import ParseResultBlock, coerce_source_span


def format_block_info(block: ParseResultBlock, index: int) -> str:
    """格式化单个解析块的信息用于显示"""
    return f"""
---
Block #{index}:
  Type: {block.block_type}
  Text: {block.text[:100]}{'...' if len(block.text) > 100 else ''}
  Page: {block.page_no}
  Section: {' > '.join(block.section_path) if block.section_path else 'N/A'}
  Order: {block.order}
  Source Span: Lines {block.source_span.start_line}-{block.source_span.end_line if block.source_span else 'N/A'}
  Metadata: {block.metadata}
"""


def test_markdown_parser():
    """测试 Markdown 解析器"""
    print("=" * 60)
    print("🔍 测试 Markdown 解析器")
    print("=" * 60)

    md_file = Path(__file__).parent / "并发编程-锁.md"
    if not md_file.exists():
        print(f"❌ 文件不存在: {md_file}")
        return

    print(f"📄 测试文件: {md_file.name}")
    print(f"📊 文件大小: {md_file.stat().st_size} bytes")
    print()

    try:
        # 构建 Markdown 解析器
        parser = build_parser("md")

        # 解析文档
        parse_config = {
            "file_type": "md",
            "file_path": str(md_file),
            "doc_name": md_file.name
        }

        doc_id = "test_md_001"
        raw_blocks = parser.parse(doc_id, parse_config)

        print(f"✅ 解析成功！共解析出 {len(raw_blocks)} 个块\n")

        # 转换为 ParseResultBlock 对象并显示前几个块
        blocks: list[ParseResultBlock] = []
        for block_dict in raw_blocks:
            source_span = coerce_source_span(block_dict.get("source_span"))
            block = ParseResultBlock(
                text=block_dict.get("text", ""),
                block_type=block_dict.get("block_type", ""),
                page_no=block_dict.get("page_no"),
                bbox=block_dict.get("bbox"),
                section_path=block_dict.get("section_path", []),
                order=block_dict.get("order", 0),
                source_span=source_span,
                metadata=block_dict.get("metadata", {}),
            )
            blocks.append(block)

        # 显示前 5 个块的详细信息
        show_count = min(5, len(blocks))
        print(f"📋 前 {show_count} 个块详细信息:")
        for i, block in enumerate(blocks[:show_count], 1):
            print(format_block_info(block, i))

        # 统计信息
        print("\n📊 统计信息:")
        print(f"  总块数: {len(blocks)}")
        print(f"  总字符数: {sum(len(block.text) for block in blocks)}")

        # 按类型统计
        type_counts = {}
        for block in blocks:
            type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1

        print(f"  块类型分布:")
        for block_type, count in sorted(type_counts.items()):
            print(f"    - {block_type}: {count}")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()


def test_pdf_parser():
    """测试 PDF 解析器"""
    print("\n" + "=" * 60)
    print("🔍 测试 PDF 解析器")
    print("=" * 60)

    pdf_file = Path(__file__).parent / "简历.pdf"
    if not pdf_file.exists():
        print(f"❌ 文件不存在: {pdf_file}")
        return

    print(f"📄 测试文件: {pdf_file.name}")
    print(f"📊 文件大小: {pdf_file.stat().st_size} bytes")
    print()

    try:
        # 构建 PDF 解析器
        parser = build_parser("pdf")

        # 解析文档
        parse_config = {
            "file_type": "pdf",
            "file_path": str(pdf_file),
            "doc_name": pdf_file.name
        }

        doc_id = "test_pdf_001"
        raw_blocks = parser.parse(doc_id, parse_config)

        print(f"✅ 解析成功！共解析出 {len(raw_blocks)} 个块\n")

        # 转换为 ParseResultBlock 对象并显示前几个块
        blocks: list[ParseResultBlock] = []
        for block_dict in raw_blocks:
            source_span = coerce_source_span(block_dict.get("source_span"))
            block = ParseResultBlock(
                text=block_dict.get("text", ""),
                block_type=block_dict.get("block_type", ""),
                page_no=block_dict.get("page_no"),
                bbox=block_dict.get("bbox"),
                section_path=block_dict.get("section_path", []),
                order=block_dict.get("order", 0),
                source_span=source_span,
                metadata=block_dict.get("metadata", {}),
            )
            blocks.append(block)

        # 显示前 5 个块的详细信息
        show_count = min(5, len(blocks))
        print(f"📋 前 {show_count} 个块详细信息:")
        for i, block in enumerate(blocks[:show_count], 1):
            print(format_block_info(block, i))

        # 统计信息
        print("\n📊 统计信息:")
        print(f"  总块数: {len(blocks)}")
        print(f"  总字符数: {sum(len(block.text) for block in blocks)}")

        # 按页码统计
        page_counts = {}
        for block in blocks:
            if block.page_no:
                page_counts[block.page_no] = page_counts.get(block.page_no, 0) + 1

        print(f"  页码分布:")
        for page_no in sorted(page_counts.keys()):
            print(f"    - 第 {page_no} 页: {page_counts[page_no]} 个块")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主测试函数"""
    print("🚀 开始测试 Parsing 模块")
    print()

    # 测试 Markdown 解析器
    test_markdown_parser()

    # 测试 PDF 解析器
    test_pdf_parser()

    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()