#!/usr/bin/env python3
"""
详细的解析结果展示脚本
生成更详细的解析结果报告
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


def generate_detailed_report():
    """生成详细的解析报告"""
    print("🎯 生成详细解析报告...")
    print()

    # 创建输出文件
    output_file = Path(__file__).parent / f"parsing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入报告头
        f.write("=" * 80 + "\n")
        f.write("Parsing 模块详细解析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # === Markdown 解析报告 ===
        f.write("📄 MARKDOWN 解析报告\n")
        f.write("-" * 80 + "\n\n")

        md_file = Path(__file__).parent / "并发编程-锁.md"
        if md_file.exists():
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

                # 写入总体统计
                f.write(f"文件: {md_file.name}\n")
                f.write(f"文件大小: {md_file.stat().st_size} bytes\n")
                f.write(f"解析块数: {len(blocks)}\n")
                f.write(f"总字符数: {sum(len(block.text) for block in blocks)}\n\n")

                # 类型分布
                type_counts = {}
                for block in blocks:
                    type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1

                f.write("块类型分布:\n")
                for block_type, count in sorted(type_counts.items()):
                    f.write(f"  - {block_type}: {count}\n")
                f.write("\n")

                # 章节结构
                f.write("章节结构:\n")
                sections = set()
                for block in blocks:
                    if block.section_path:
                        sections.add(" > ".join(block.section_path))
                for section in sorted(sections):
                    f.write(f"  - {section}\n")
                f.write("\n")

                # 详细块内容 (显示前10个)
                f.write("前10个块详细内容:\n")
                f.write("-" * 80 + "\n")
                for i, block in enumerate(blocks[:10], 1):
                    f.write(f"\n块 #{i} [{block.block_type}]\n")
                    f.write(f"  章节: {' > '.join(block.section_path) if block.section_path else 'N/A'}\n")
                    f.write(f"  顺序: {block.order}\n")
                    f.write(f"  内容: {block.text}\n")
                    f.write(f"  来源: 行 {block.source_span.start_line}-{block.source_span.end_line if block.source_span else 'N/A'}\n")

            except Exception as e:
                f.write(f"❌ 解析失败: {e}\n")

        else:
            f.write(f"❌ 文件不存在: {md_file}\n")

        # === PDF 解析报告 ===
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("📄 PDF 解析报告\n")
        f.write("-" * 80 + "\n\n")

        pdf_file = Path(__file__).parent / "简历.pdf"
        if pdf_file.exists():
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

                # 写入总体统计
                f.write(f"文件: {pdf_file.name}\n")
                f.write(f"文件大小: {pdf_file.stat().st_size} bytes\n")
                f.write(f"解析块数: {len(blocks)}\n")
                f.write(f"总字符数: {sum(len(block.text) for block in blocks)}\n\n")

                # 页码分布
                page_counts = {}
                for block in blocks:
                    if block.page_no:
                        page_counts[block.page_no] = page_counts.get(block.page_no, 0) + 1

                f.write("页码分布:\n")
                for page_no in sorted(page_counts.keys()):
                    f.write(f"  - 第 {page_no} 页: {page_counts[page_no]} 个块\n")
                f.write("\n")

                # 详细块内容 (显示前10个)
                f.write("前10个块详细内容:\n")
                f.write("-" * 80 + "\n")
                for i, block in enumerate(blocks[:10], 1):
                    f.write(f"\n块 #{i} [{block.block_type}]\n")
                    f.write(f"  页码: {block.page_no}\n")
                    f.write(f"  章节: {' > '.join(block.section_path) if block.section_path else 'N/A'}\n")
                    f.write(f"  顺序: {block.order}\n")
                    f.write(f"  内容: {block.text}\n")
                    f.write(f"  来源: 页码 {block.source_span.page_no if block.source_span else 'N/A'}\n")

            except Exception as e:
                f.write(f"❌ 解析失败: {e}\n")

        else:
            f.write(f"❌ 文件不存在: {pdf_file}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")

    print(f"✅ 报告已生成: {output_file}")
    print(f"📊 总共生成解析报告包含:")
    print(f"  - Markdown 解析详情")
    print(f"  - PDF 解析详情")
    print(f"  - 统计信息和结构分析")
    print()

    # 显示报告的前几行预览
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        preview_lines = 20
        print(f"📋 报告预览 (前 {preview_lines} 行):")
        print("=" * 80)
        for line in lines[:preview_lines]:
            print(line.rstrip())


if __name__ == "__main__":
    generate_detailed_report()