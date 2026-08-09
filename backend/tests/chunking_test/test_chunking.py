#!/usr/bin/env python3
"""
Chunking 模块集成测试（tests/chunking_test/）

与 parsing_test 平行的测试目录：
  tests/parsing_test/    → 解析测试（md + pdf）
  tests/chunking_test/   → 分块测试（本目录）

将 parsing 阶段的输出作为数据源，演示 MarkdownChunker 的父子双粒度分块效果。
测试数据文件与 parsing_test 共用（同一份文档贯穿 parsing → chunking 流水线，
保持数据单一来源，见下方 _DATA_DIR）。
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.src.parsing.parser_factory import build_parser
from backend.src.parsing.models import ParseResultBlock, coerce_source_span
from backend.src.chunking import MarkdownChunker, ChunkConfig

# 测试数据目录：与 parsing_test 共用同一份文档（数据单一来源，不重复复制）
_DATA_DIR = Path(__file__).resolve().parent.parent / "parsing_test"


# ================================================================
# 辅助函数
# ================================================================

def _parse_and_flatten(parser_name: str, file_path: Path) -> list[dict]:
    """解析文件，返回结构化的 ParseResultBlock dict 列表。"""
    parser = build_parser(parser_name)
    raw = parser.parse(
        doc_id=f"test_{parser_name}_001",
        parse_config={
            "file_type": parser_name,
            "file_path": str(file_path),
            "doc_name": file_path.name,
        },
    )
    blocks: list[dict] = []
    for b in raw:
        span = coerce_source_span(b.get("source_span"))
        blocks.append({
            "doc_id": f"test_{parser_name}_001",
            "text": b["text"],
            "block_type": b["block_type"],
            "page_no": b.get("page_no"),
            "bbox": b.get("bbox"),
            "section_path": list(b.get("section_path", [])),
            "order": b["order"],
            "source_span": span,
            "metadata": b.get("metadata", {}),
        })
    return blocks


def _indent(text: str, spaces: int = 2) -> str:
    """给多行文本添加缩进。"""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def _truncate(text: str, max_len: int = 120) -> str:
    """截断文本并加省略号。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


# ================================================================
# Markdown 分块测试
# ================================================================

def test_markdown_chunking():
    """测试 Markdown 文件的父子分块效果"""
    print("=" * 80)
    print("📝 Markdown Chunking 测试")
    print("=" * 80)
    print()

    md_file = _DATA_DIR / "并发编程-锁.md"
    if not md_file.exists():
        print(f"❌ 文件不存在: {md_file}")
        return

    # ── Step 1: Parsing ──
    print("📖 Step 1: 解析 Markdown 文件...")
    blocks = _parse_and_flatten("md", md_file)
    print(f"   解析出 {len(blocks)} 个 block")
    print(f"   块类型分布: {_type_distribution(blocks)}")
    print()

    # ── Step 2: Chunking ──
    print("✂️  Step 2: 父子双粒度分块...")
    config = ChunkConfig(
        parent_target_tokens=512,
        parent_max_tokens=768,
        child_target_tokens=128,
        child_max_tokens=192,
    )
    chunker = MarkdownChunker()
    chunks = chunker.chunk(blocks, config)
    print(f"   切分出 {len(chunks)} 个 chunk")
    print()

    # ── Step 3: 分角色统计 ──
    parents = [c for c in chunks if c["chunk_role"] == "parent"]
    children = [c for c in chunks if c["chunk_role"] == "child"]
    print("📊 Step 3: 父子分块统计")
    print(f"   Parent 数量: {len(parents)}")
    print(f"   Child  数量: {len(children)}")
    print(f"   父子比例: 1:{len(children)/max(len(parents),1):.1f}")
    print()

    # ── 父块统计 ──
    print("   Parent token 分布:")
    for p in parents:
        tk = p["meta"]["token_count"]
        role = p["section_path"][-1] if p["section_path"] else "N/A"
        bar = _token_bar(tk, config.parent_target_tokens)
        print(f"     [{bar}] {tk:>4}t  {_truncate(role, 40)}")
    print()

    # ── 子块统计 ──
    print("   Child 切分方式:")
    split_counts: dict[str, int] = {}
    for c in children:
        by = c["meta"]["trace"].get("split_by", "unknown")
        split_counts[by] = split_counts.get(by, 0) + 1
    for method, count in sorted(split_counts.items()):
        print(f"     {method}: {count} 个")
    print()

    # ── Step 4: 展示前 6 条 chunk ──
    print("📋 Step 4: 前 6 条 chunk 详细展示")
    print("-" * 80)
    for i, c in enumerate(chunks[:6], 1):
        role_icon = "🟦" if c["chunk_role"] == "parent" else "🟩"
        print(f"\n  [{role_icon} #{i}] {c['chunk_role'].upper()}")
        print(f"  chunk_id:   {c['chunk_id']}")
        print(f"  token:      {c['meta']['token_count']}")
        print(f"  section:    {' > '.join(c['section_path'])}")
        print(f"  block_type: {c['meta']['block_types']}")
        print(f"  text:       {_truncate(c['text'], 130)}")
        if c["chunk_role"] == "parent" and c["child_ids"]:
            print(f"  children:   {c['child_ids']}")
        if c["chunk_role"] == "child" and c["parent_id"]:
            print(f"  parent:     {c['parent_id']}")


# ================================================================
# PDF 分块测试 (简要)
# ================================================================

def test_pdf_chunking():
    """测试 PDF 文件的分块效果（简要展示）"""
    print("\n" + "=" * 80)
    print("📄 PDF Chunking 测试")
    print("=" * 80)
    print()

    pdf_file = _DATA_DIR / "简历.pdf"
    if not pdf_file.exists():
        print(f"❌ 文件不存在: {pdf_file}")
        return

    # ── Step 1: Parsing ──
    print("📖 Step 1: 解析 PDF 文件...")
    blocks = _parse_and_flatten("pdf", pdf_file)
    print(f"   解析出 {len(blocks)} 个 block")
    print()

    # ── Step 2: Chunking ──
    print("✂️  Step 2: 父子双粒度分块...")
    config = ChunkConfig(
        parent_target_tokens=512,
        parent_max_tokens=768,
        child_target_tokens=128,
        child_max_tokens=192,
    )
    chunker = MarkdownChunker()
    chunks = chunker.chunk(blocks, config)
    print(f"   切分出 {len(chunks)} 个 chunk")
    print()

    # ── Step 3: 分角色统计 ──
    parents = [c for c in chunks if c["chunk_role"] == "parent"]
    children = [c for c in chunks if c["chunk_role"] == "child"]
    print("📊 Step 3: 父子分块统计")
    print(f"   Parent 数量: {len(parents)}")
    print(f"   Child  数量: {len(children)}")
    print()

    # ── 展示前 5 条 chunk ──
    print("📋 前 5 条 chunk 展示")
    print("-" * 80)
    for i, c in enumerate(chunks[:5], 1):
        role_icon = "🟦" if c["chunk_role"] == "parent" else "🟩"
        print(f"\n  [{role_icon} #{i}] {c['chunk_role'].upper()}")
        print(f"  chunk_id: {c['chunk_id']}")
        print(f"  token:    {c['meta']['token_count']}")
        print(f"  page_no:  {c['page_no']}")
        print(f"  text:     {_truncate(c['text'], 120)}")


# ================================================================
# 工具
# ================================================================

def _type_distribution(blocks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in blocks:
        t = b.get("block_type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def _token_bar(tokens: int, target: int) -> str:
    """生成 token 进度条。"""
    ratio = min(tokens / target, 1.5)
    filled = int(ratio * 20)
    bar = "█" * min(filled, 20) + "░" * max(20 - filled, 0)
    if tokens > target * 1.2:
        return bar + "  OVER"
    return bar


# ================================================================
# 主入口
# ================================================================

def main():
    print("🚀 Chunking 模块集成测试")
    print()

    # 测试 Markdown 分块（详细）
    test_markdown_chunking()

    # 测试 PDF 分块（简要）
    test_pdf_chunking()

    print("\n" + "=" * 80)
    print("✅ Chunking 测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
