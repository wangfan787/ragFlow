#!/usr/bin/env python3
"""chunking 全流程 walkthrough demo（适合理解整套流水线）。

用一份精心设计的简洁 Markdown（sample_walkthrough.md）跑通：
  原始 Markdown → ① parsing 解析成 block → ② 合并父块 → ③ 切子块 → ④ 最终 chunk
每一步都把"解析出的内容"完整打印出来，字段逐条注释。

运行（务必用 conda agent 环境）：
  /home/xvxing/miniconda3/envs/agent/bin/python backend/tests/walkthrough_demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.parsing.parser_factory import build_parser
from backend.src.parsing.models import coerce_source_span
from backend.src.chunking import MarkdownChunker, ChunkConfig

MD_FILE = PROJECT_ROOT / "backend" / "tests" / "parsing_test" / "sample_walkthrough.md"
DOC_ID = "sample"
AGENT_PY_HINT = "/home/xvxing/miniconda3/envs/agent/bin/python"

# 块类型 → 显示图标
_ICON = {
    "frontmatter": "📋", "heading": "🏷️ ", "paragraph": "📝", "table": "📊",
    "list": "📑", "code": "💻", "blockquote": "💬", "hr": "➖",
}


def _oneline(s: str, n: int = 64) -> str:
    """把多行文本压成一行（换行显示为 ⏎），并截断。"""
    s = s.replace("\r\n", "\n").replace("\n", "⏎").strip()
    return s if len(s) <= n else s[:n] + "…"


def _bar(tokens: int, target: int, width: int = 20) -> str:
    """生成 token 占比进度条（用于父块体量可视化）。"""
    filled = min(int(tokens / target * width), width)
    return "█" * filled + "░" * (width - filled)


def parse_md() -> tuple[list[dict], list[dict]]:
    """Step A：调用 MarkdownParser 把 md 解析成 block 列表。

    返回 (raw_blocks, chunk_blocks)：
      - raw_blocks：parser 的【原始】输出，dict 完整字段，未做任何改造。
                    用来看「parser 到底吐了什么」。
      - chunk_blocks：在 raw 基础上补 doc_id、把 source_span 转成对象的版本。
                      chunker 需要这个形态（它读 doc_id），用于后续 Step 2~4。
    两者是同一批数据，区别仅在本 demo 做的二次封装。
    """
    parser = build_parser("md")
    raw = parser.parse(
        doc_id=DOC_ID,
        parse_config={"file_type": "md", "file_path": str(MD_FILE), "doc_name": MD_FILE.name},
    )
    raw_blocks = [dict(b) for b in raw]  # 浅拷贝，避免下游改动污染原始样本
    chunk_blocks = []
    for b in raw:
        chunk_blocks.append({
            "doc_id": DOC_ID,
            "text": b["text"],
            "block_type": b["block_type"],
            "section_path": list(b.get("section_path", [])),
            "order": b["order"],
            "page_no": b.get("page_no"),
            "source_span": coerce_source_span(b.get("source_span")),
        })
    return raw_blocks, chunk_blocks


def print_raw_md() -> None:
    W = 92
    print("=" * W)
    print("原始 Markdown 文件（sample_walkthrough.md）")
    print("=" * W)
    for i, line in enumerate(MD_FILE.read_text(encoding="utf-8").splitlines(), 1):
        print(f"  L{i:>2}│ {line}")
    print()


def print_parsing(raw_blocks: list[dict], blocks: list[dict]) -> None:
    import json
    W = 92
    print("=" * W)
    print("【Step 1】parsing：Markdown → 结构化 block 列表")
    print("=" * W)

    # ── 1a. parser 原始输出：每个 block 的完整 dict（8 个字段全展开）──
    print("【1a】MarkdownParser.parse() 的原始返回 —— 每个 block 的完整 dict")
    print("-" * W)
    print("下面是 parser【未经任何改造】的输出。每个 block 都是同样 8 个字段的 dict：")
    print("  text(内容) / block_type(类型) / order(顺序) / section_path(章节路径)")
    print("  source_span(原文行/字符范围) / page_no / bbox / metadata")
    print("  （page_no/bbox/metadata 在 Markdown 解析下恒为 null/空，供 PDF 解析复用同一结构）\n")
    for b in raw_blocks:
        print(f"  ── block #{b['order']} [{b['block_type']}] 的完整 dict ──")
        for line in json.dumps(b, ensure_ascii=False, indent=2).splitlines():
            print(f"  {line}")
        print()

    # ── 1b. 精简视图（按类型聚合，便于快速浏览）──
    print("【1b】精简视图（同上数据，按出现顺序一览）")
    print("-" * W)
    print("（这版只展示关键字段，方便快速对照；完整字段见上方 1a）")
    print(f"共解析出 {len(blocks)} 个 block：\n")
    for b in blocks:
        icon = _ICON.get(b["block_type"], "  ")
        sp = b["source_span"]
        print(f"  block #{b['order']:<2} {icon} [{b['block_type']:<11}]")
        print(f"           章节: {' > '.join(b['section_path']) or '（文档级，无章节）'}")
        print(f"           行号: L{sp.start_line}–L{sp.end_line}")
        print(f"           内容: {_oneline(b['text'], 78)}")
    # 类型分布统计
    dist: dict[str, int] = {}
    for b in blocks:
        dist[b["block_type"]] = dist.get(b["block_type"], 0) + 1
    print(f"\n  📊 block 类型分布: {dist}")
    print()


def print_parents(blocks: list[dict], config: ChunkConfig) -> list[dict]:
    W = 92
    chunker = MarkdownChunker()
    parents = chunker._merge._merge_into_parents(blocks, config)


    
    print("=" * W)
    print("【Step 2】阶段一：把 block 合并成「父块」(目标 ~%dtoken)" % config.parent_target_tokens)
    print("=" * W)
    print("规则：用 buffer 逐 block 累积，遇到以下情况就把 buffer 写成一个父块（flush）：")
    print("  • 单 block 超 768token(硬上限) → 单独成父块")
    print("  • 遇到 code/table → 先 flush 已有内容，代码/表格单独成父块（不拆散）")
    print("  • 遇到 heading/frontmatter/hr(语义边界) → flush")
    print("  • 累积超 512token(软上限) → flush\n")
    print(f"本样本合并出 {len(parents)} 个父块：\n")
    for i, p in enumerate(parents, 1):
        tk = chunker._counter.count(p["text"])
        nblk = len(p["source_blocks"])
        bnd = p.get("trace", {}).get("parent_boundary")
        long_flag = p.get("trace", {}).get("split_long_block", False)
        reason = ("超长block单独成块" if long_flag
                  else (f"heading边界flush" if bnd == "heading"
                        else (f"{bnd}边界flush" if bnd else "buffer累积/末尾flush")))
        print(f"  父块#{i}  [{_bar(tk, config.parent_target_tokens)}] ~{tk:>3}t  "
              f"含 {nblk} 个block  切因: {reason}")
        print(f"          内容: {_oneline(p['text'], 82)}")
    print()
    return parents


def print_children(parents: list[dict], config: ChunkConfig) -> None:
    W = 92
    chunker = MarkdownChunker()
    merge = chunker._merge
    print("=" * W)
    print(f"【Step 3】阶段二：每个父块内部切成「子块」(目标 ~{config.child_target_tokens}token)")
    print("=" * W)
    print("规则：按 block 类型分别处理——")
    print("  • heading      → 单独成子块（召回价值高）")
    print("  • code/table   → 整体一个子块（不拆）")
    print("  • 段落 ≤128t   → 整段一个子块")
    print("  • 段落 >128t   → 按句号/分号/换行切句 + 贪心打包")
    print()
    for i, p in enumerate(parents, 1):
        children = merge._split_into_children(p, config)
        print(f"  父块#{i} → 切出 {len(children)} 个子块:")
        for j, c in enumerate(children, 1):
            tk = chunker._counter.count(c["text"])
            print(f"      子块{i}.{j}  split_by={c['trace']['split_by']:<9} ~{tk:>3}t  "
                  f"{_oneline(c['text'], 52)}")
    print()


def print_final_chunks(blocks: list[dict], config: ChunkConfig) -> None:
    W = 92
    chunker = MarkdownChunker()
    chunks = chunker.chunk(blocks, config)
    n_p = sum(1 for c in chunks if c["chunk_role"] == "parent")
    n_c = sum(1 for c in chunks if c["chunk_role"] == "child")

    print("=" * W)
    print("【Step 4】最终输出：MarkdownChunker.chunk() → 扁平 chunk 列表")
    print("=" * W)
    print(f"三遍扫描：① 分配 chunk_id ② 回填 parent_id ③ 组装 ChunkRecord + ChunkMeta")
    print(f"共 {len(chunks)} 个 chunk（父块 {n_p}，子块 {n_c}），深度优先排列（父块紧跟其子块）。\n")

    # ── 全量列表（紧凑视图）──
    print("── 全部 chunk 一览（父子关系树）──\n")
    for c in chunks:
        icon = "🟦" if c["chunk_role"] == "parent" else "  🟩"
        indent = "" if c["chunk_role"] == "parent" else "      "
        tk = c["meta"]["token_count"]
        print(f"  {indent}{icon} {c['chunk_id']:<18} ~{tk:>3}t  {_oneline(c['text'], 46)}")
    print()

    # ── 选一条多 block 父块 + 它的子块，逐字段拆解 ──
    multi_parent = next(
        (c for c in chunks if c["chunk_role"] == "parent" and len(c["meta"]["source_block_ids"]) >= 2),
        next(c for c in chunks if c["chunk_role"] == "parent"),
    )
    print("── 字段逐条拆解（选一条含多个 block 的父块 + 其首个子块）──")
    print(f"   选用: 父块 {multi_parent['chunk_id']}（含 {len(multi_parent['meta']['source_block_ids'])} 个源block）\n")

    def dissect(title: str, c: dict) -> None:
        m = c["meta"]
        print(f"  ▶ {title}  {c['chunk_id']}")
        print(f"     ── 基本身份 ──")
        print(f"     chunk_id         {c['chunk_id']:<22} 全局唯一ID，格式 {{doc_id}}_ck_{{order}}")
        print(f"     doc_id           {c['doc_id']:<22} 所属文档")
        print(f"     chunk_role       {c['chunk_role']:<22} parent=完整上下文 / child=精准召回")
        print(f"     chunk_order      {str(c['chunk_order']):<22} 文档内全局顺序号")
        print(f"     ── 位置与路径 ──")
        print(f"     section_path     {' > '.join(c['section_path']):<22} 章节面包屑，取自首个源block")
        print(f"     page_no          {str(c['page_no']):<22} PDF才有，Markdown恒为None")
        print(f"     ── 溯源信息 ──")
        print(f"     block_types      {str(m['block_types']):<22} 源block类型(不去重,可重复)")
        print(f"     source_block_ids {str(m['source_block_ids'])}")
        print(f"                      {'':<22} 源block的ID(去重)，格式 {{doc_id}}_blk_{{order}}")
        print(f"     source_span      行{m['source_span']['start_line']}–{m['source_span']['end_line']}"
              f"{'':>14} 原文行号范围(多block合并取最小~最大)")
        print(f"     ── 体量 ──")
        print(f"     token_count      {str(m['token_count']):<22} tiktoken cl100k_base 计数")
        print(f"     ── 父子关系(Small-to-Big 核心) ──")
        print(f"     parent_id        {str(c['parent_id']):<22} 子块→父块；父块为None")
        print(f"     child_ids        {str(c['child_ids'])}")
        print(f"                      {'':<22} 父块→子块列表；子块为空")
        print(f"     ── 切分追踪 ──")
        print(f"     trace            {str(m['trace']):<22} 记录「为何这么切」")
        print(f"     ── 文本 ──")
        print(f"     text             {_oneline(c['text'], 60)}")
        print()

    dissect("父块", multi_parent)
    child_id = multi_parent["child_ids"][0] if multi_parent["child_ids"] else None
    if child_id:
        child = next(c for c in chunks if c["chunk_id"] == child_id)
        dissect("其首子块", child)

    # ── 检索流程演示 ──
    print("── Small-to-Big 检索流程演示 ──\n")
    if child_id:
        print(f"  用户 query: 「自旋锁 加锁」")
        print(f"  1️⃣  向量检索命中子块 {child_id}（~{child['meta']['token_count']}t，精准）")
        print(f"  2️⃣  读 child.parent_id → {child['parent_id']}")
        print(f"  3️⃣  回溯父块 {multi_parent['chunk_id']}（~{multi_parent['meta']['token_count']}t，完整上下文）")
        print(f"  4️⃣  返回父块文本 → 用户拿到完整章节，而非半句话\n")

    print("=" * W)
    print("✅ walkthrough 完成。")
    print("   一句话总结：parsing 拆 block → 父块给上下文、子块给精准召回 →")
    print("   命中子块回溯父块，天然覆盖跨块边界，无需 overlap 拼接。")
    print("=" * W)


def main() -> None:
    config = ChunkConfig(
        parent_target_tokens=512, parent_max_tokens=768,
        child_target_tokens=128, child_max_tokens=192,
    )
    print_raw_md()
    raw_blocks, blocks = parse_md()
    print_parsing(raw_blocks, blocks)
    parents = print_parents(blocks, config)
    print_children(parents, config)
    print_final_chunks(blocks, config)


if __name__ == "__main__":
    main()
