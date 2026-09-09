#!/usr/bin/env python3
"""
完整解析结果展示脚本 - 展示所有解析块
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径到 Python 模块搜索路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.src.parsing.parser_factory import build_parser
from backend.src.parsing.models import ParseResultBlock, coerce_source_span


def show_full_markdown_results():
    """展示完整的 Markdown 解析结果"""
    print("=" * 80)
    print("📄 完整 Markdown 解析结果")
    print("=" * 80)
    print()

    md_file = Path(__file__).parents[2] / "tests" / "parsing" / "data" / "并发编程-锁.md"
    if not md_file.exists():
        print(f"❌ 文件不存在: {md_file}")
        return

    parser = build_parser("md")
    parse_config = {
        "file_type": "md",
        "file_path": str(md_file),
        "doc_name": md_file.name
    }

    try:
        raw_blocks = parser.parse("test_md_001", parse_config)
        blocks = []
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

        print(f"📊 基本信息:")
        print(f"  文件: {md_file.name}")
        print(f"  文件大小: {md_file.stat().st_size} bytes")
        print(f"  解析块数: {len(blocks)}")
        print(f"  总字符数: {sum(len(block.text) for block in blocks)}")
        print()

        # 类型分布
        type_counts = {}
        for block in blocks:
            type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1

        print(f"📈 块类型分布:")
        for block_type, count in sorted(type_counts.items()):
            print(f"  - {block_type}: {count}")
        print()

        # 完整章节结构
        sections = set()
        for block in blocks:
            if block.section_path:
                sections.add(" > ".join(block.section_path))

        print(f"🗂️ 完整章节结构:")
        for section in sorted(sections):
            print(f"  - {section}")
        print()

        # 所有块的详细信息
        print(f"📋 所有 {len(blocks)} 个块详细信息:")
        print("=" * 80)

        for i, block in enumerate(blocks, 1):
            print(f"\n块 #{i} [{block.block_type}]")
            print(f"  章节: {' > '.join(block.section_path) if block.section_path else 'N/A'}")
            print(f"  顺序: {block.order}")
            print(f"  来源: 行 {block.source_span.start_line}-{block.source_span.end_line if block.source_span else 'N/A'}")

            # 显示完整文本内容
            text_lines = block.text.split('\n')
            if len(text_lines) > 5:
                print(f"  内容预览 (前5行共{len(text_lines)}行):")
                for line in text_lines[:5]:
                    print(f"    {line}")
                if len(text_lines) > 5:
                    print(f"    ... (还有 {len(text_lines) - 5} 行)")
            else:
                print(f"  内容 ({len(text_lines)}行):")
                for line in text_lines:
                    print(f"    {line}")

        print("\n" + "=" * 80)
        print("✅ Markdown 解析完成")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()


def show_full_pdf_results():
    """展示完整的 PDF 解析结果"""
    print("\n" + "=" * 80)
    print("📄 完整 PDF 解析结果")
    print("=" * 80)
    print()

    pdf_file = Path(__file__).parents[2] / "tests" / "parsing" / "data" / "简历.pdf"
    if not pdf_file.exists():
        print(f"❌ 文件不存在: {pdf_file}")
        return

    parser = build_parser("pdf")
    parse_config = {
        "file_type": "pdf",
        "file_path": str(pdf_file),
        "doc_name": pdf_file.name
    }

    try:
        raw_blocks = parser.parse("test_pdf_001", parse_config)
        blocks = []
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

        print(f"📊 基本信息:")
        print(f"  文件: {pdf_file.name}")
        print(f"  文件大小: {pdf_file.stat().st_size} bytes")
        print(f"  解析块数: {len(blocks)}")
        print(f"  总字符数: {sum(len(block.text) for block in blocks)}")
        print()

        # 页码分布
        page_counts = {}
        for block in blocks:
            if block.page_no:
                page_counts[block.page_no] = page_counts.get(block.page_no, 0) + 1

        print(f"📈 页码分布:")
        for page_no in sorted(page_counts.keys()):
            print(f"  - 第 {page_no} 页: {page_counts[page_no]} 个块")
        print()

        # 所有块的详细信息
        print(f"📋 所有 {len(blocks)} 个块详细信息:")
        print("=" * 80)

        for i, block in enumerate(blocks, 1):
            print(f"\n块 #{i} [{block.block_type}]")
            print(f"  页码: {block.page_no}")
            print(f"  章节: {' > '.join(block.section_path) if block.section_path else 'N/A'}")
            print(f"  顺序: {block.order}")
            print(f"  来源: 页码 {block.source_span.page_no if block.source_span else 'N/A'}")

            # 显示文本内容
            text = block.text.strip()
            if len(text) > 100:
                print(f"  内容预览: {text[:100]}...")
            else:
                print(f"  内容: {text}")

        print("\n" + "=" * 80)
        print("✅ PDF 解析完成")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🚀 完整解析结果展示")
    print()

    # 展示完整 Markdown 结果
    show_full_markdown_results()

    # 展示完整 PDF 结果
    show_full_pdf_results()

    print("\n" + "=" * 80)
    print("🎉 所有解析结果展示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()