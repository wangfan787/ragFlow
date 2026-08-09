"""两阶段父子分块合并策略：BlockMergeStrategy。

================================================================================
工作流概述
================================================================================

输入：parsing 阶段的 ParseResultBlock 列表（结构化 block）
输出：父块 + 子块混合列表（扁平，深度优先）

两阶段流程：
  阶段一 _merge_into_parents()
    ┌─────────┐  ┌──────────┐  ┌────────┐    ┌───────────┐
    │ heading  │  │ paragraph│  │  table │ →  │ Parent #1 │ ~512 tokens
    └─────────┘  └──────────┘  └────────┘    └───────────┘
    ┌─────────┐  ┌──────────┐                ┌───────────┐
    │ heading  │  │   code   │             →  │ Parent #2 │ ~512 tokens
    └─────────┘  └──────────┘                └───────────┘

  阶段二 _split_into_children() — 对每个父块内部切子块
    ┌───────────┐
    │ Parent #1  │
    └─────┬─────┘
          ├─→ Child #1.1 (heading)       ~10 tokens
          ├─→ Child #1.2 (paragraph)     ~128 tokens
          └─→ Child #1.3 (table)          ~80 tokens

================================================================================
设计原则
================================================================================

1. 语义边界优先：heading/frontmatter/hr 是天然的主题分界，遇到它们先 flush 已有父块。
2. 特殊块不拆：code 和 table 整体保留，不做句子级切分，保持语义完整。
3. 句子级切分：超长段落先按中英文标点 + 换行切为句子，再贪心打包。
4. 父子关系覆盖跨块上下文：不再使用文本拼接 overlap，命中子块后回溯父块。
5. 硬/软双上限：target_tokens 是期望值（软），max_tokens 是绝对不能超过的值（硬），
   给贪心打包算法一个缓冲区。

================================================================================
关键数据结构
================================================================================

merge() 方法返回的扁平列表，每项 dict：

  Parent:
    {
      "text": "...",                        # 父块文本（子块文本拼接后再拼装）
      "source_blocks": [block, ...],        # 组成此父块的原始 block 列表
      "trace": {"chunk_role": "parent",     # 切分追踪（记录为何在此处 flush）
                "parent_boundary": "heading" | "paragraph" | None},
      "chunk_role": "parent",
    }

  Child:
    {
      "text": "...",                        # 子块文本
      "source_blocks": [block],             # 子块来源（通常单个 block，超长时会拆分）
      "trace": {"chunk_role": "child",      # 切分追踪（记录切分方式）
                "split_by": "heading" | "paragraph" | "sentence" | "code" | "table"},
      "chunk_role": "child",
      "parent_ref": <parent item>,          # 临时引用，MarkdownChunker 回填 parent_id
    }
"""

from __future__ import annotations

import re

from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.chunking.token_counter import SimpleTokenCounter


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 父块切分的语义边界 block_type：遇到这些 block 时，当前父块 buffer 强制 flush。
# heading 是天然主题边界，frontmatter/hr 也是文档结构分界。
_PARENT_BOUNDARY_TYPES = {"heading", "frontmatter", "hr"}


# ---------------------------------------------------------------------------
# BlockMergeStrategy
# ---------------------------------------------------------------------------

class BlockMergeStrategy:
    """两阶段父子分块策略。

    阶段一 ``_merge_into_parents``：按 heading/段落边界切父块（~512 token），
    负责完整上下文。

    阶段二 ``_split_into_children``：把每个父块内部按 heading/code/table/段落/句子
    切成子块（~128 token），负责精准召回。

    不再做文本拼接式 overlap：相邻父块之间没有共享尾部文本，子块在父块内部自包含，
    跨边界内容由父子关系天然覆盖（命中子块 → 回溯父块拿完整上下文）。
    """

    def __init__(self) -> None:
        """初始化合并策略，创建 token 计数器实例。"""
        self._counter = SimpleTokenCounter()

    # ==================================================================
    # 基础工具方法
    # ==================================================================

    def _read(self, block, key: str, default=None):
        """安全读取 block 的字段值（兼容 dict 和对象两种格式）。"""
        if isinstance(block, dict):
            return block.get(key, default)
        return getattr(block, key, default)

    def _block_type(self, block) -> str:
        """获取 block 的类型字符串，未指定时默认 "paragraph"。"""
        return str(self._read(block, "block_type", "paragraph"))

    def _preserve_whole(self, block, config: ChunkConfig) -> bool:
        """判断当前 block 是否应该整体保留不拆（code/table + 配置开关）。"""
        block_type = self._block_type(block)
        return (block_type == "code" and config.preserve_code_block) or (
            block_type == "table" and config.preserve_table_block
        )

    # ==================================================================
    # 句子切分工具
    # ==================================================================
    # 原来的实现用 text.split() 分词，对中文（无空格）完全不切，导致超长中文段
    # 无法被拆成子块。当前实现按中英文句末标点 + 换行切分，解决问题。
    # ==================================================================

    # 按中英文句末标点 + 换行切分，保留分隔符在句尾（lookahead 保留标点）。
    _SENTENCE_END_RE = re.compile(r"(?<=[。！？!?；;\n])")

    def _split_by_sentence(self, text: str) -> list[str]:
        """按句子边界切分文本，返回非空句子列表（标点保留在句尾）。

        支持中文标点（。！？；）、英文标点（.!?;）和换行（\\n）。
        不再使用空格分词，避免无空格中文长段无法被切分的问题。

        Examples:
            >>> _split_by_sentence("你好。世界！")
            ["你好。", "世界！"]

            >>> _split_by_sentence("Hello. World!")
            ["Hello.", "World!"]
        """
        cleaned = text.strip()
        if not cleaned:
            return []
        parts = self._SENTENCE_END_RE.split(cleaned)
        sentences: list[str] = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                sentences.append(stripped)
        return sentences

    def _greedy_pack_sentences(
        self, sentences: list[str], target_tokens: int, max_tokens: int
    ) -> list[str]:
        """贪心把句子打包成片段，每个片段不超过 target（软上限）/max（硬上限）token。

        算法：
        - 逐句累积到 buffer 中。
        - 加上当前句后 ≥ max_tokens → flush buffer（硬上限保护，绝不超）。
        - 累积 token 数 ≥ target_tokens → flush buffer（软上限，达到预期大小就输出）。
        - 循环结束后 flush 剩余内容。

        这样可以产出尽可能接近 target_tokens 大小的片段，
        同时保证单个句子不会被拆成两个片段。
        """
        if not sentences:
            return []
        packed: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in sentences:
            sent_tokens = max(1, self._counter.count(sentence))
            # 硬上限：当前 buffer 不为空，且加了这句会超过 max → 先 flush
            if buffer and buffer_tokens + sent_tokens > max_tokens:
                packed.append("\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            buffer.append(sentence)
            buffer_tokens += sent_tokens
            # 软上限：达到 target → flush
            if buffer_tokens >= target_tokens:
                packed.append("\n".join(buffer))
                buffer = []
                buffer_tokens = 0
        # 兜底：flush 剩余内容
        if buffer:
            packed.append("\n".join(buffer))
        return packed

    # ==================================================================
    # 阶段一：父块切分 (_merge_into_parents)
    # ==================================================================
    # 按语义边界把 blocks 累积成 ~512 token 的父块。
    #
    # Flush 触发条件（优先级从高到低）：
    #   1. 单 block token > parent_max_tokens   → 超长 block 单独成父块
    #   2. 当前 block 是 code/table且已有缓冲   → 先 flush 已有内容
    #   3. 当前 block 是 heading/frontmatter/hr → 语义边界 flush
    #   4. buffer + 当前 block > target_tokens  → 软上限 flush
    #   5. buffer >= max_tokens                 → 硬上限 flush
    # ==================================================================

    def _merge_into_parents(self, blocks: list, config: ChunkConfig) -> list[dict]:
        """按 heading/段落边界累积成父块。

        父块合并规则：
        - 遇到 heading/frontmatter/hr 强制 flush（语义边界），flush 后才加入 heading。
        - code/table 整体保留，先 flush 已有内容，再单独成一个父块（且不和前文混排）。
        - 达到 parent_target_tokens（软上限）预 flush。
        - 达到 parent_max_tokens（硬上限）强制 flush。
        """
        parents: list[dict] = []
        current_texts: list[str] = []
        current_blocks: list = []
        current_tokens = 0

        def flush(boundary: str | None = None) -> None:
            """将当前累积的文本和 block 作为一个父块写出。"""
            nonlocal current_texts, current_blocks, current_tokens
            if not current_texts:
                return
            trace: dict = {}
            if boundary:
                # 记录触发 flush 的边界类型，便于调试和日志追踪
                trace["parent_boundary"] = boundary
            parents.append(
                {
                    "text": "\n\n".join(current_texts).strip(),
                    "source_blocks": list(current_blocks),
                    "trace": trace,
                }
            )
            # 重置 buffer
            current_texts = []
            current_blocks = []
            current_tokens = 0

        for block in blocks:
            # 获取 block 文本（跳过空文本）
            text = str(self._read(block, "text", "")).strip()
            if not text:
                continue

            block_type = self._block_type(block)
            block_tokens = max(1, self._counter.count(text))
            preserve_whole = self._preserve_whole(block, config)
            is_boundary = block_type in _PARENT_BOUNDARY_TYPES

            # ---- 规则 2：code/table 先 flush 已有内容 ----
            if preserve_whole and current_texts:
                flush()

            # ---- 规则 1：超长 block 单独成父块 ----
            if block_tokens > config.parent_max_tokens:
                flush()
                parents.append(
                    {
                        "text": text,
                        "source_blocks": [block],
                        "trace": {"split_long_block": True},
                    }
                )
                continue

            # ---- 规则 2 补充：preserve 的 code/table 单独成父块 ----
            if preserve_whole:
                parents.append({"text": text, "source_blocks": [block]})
                continue

            # ---- 规则 3：语义边界 flush ----
            if is_boundary and current_texts:
                flush(boundary=block_type)

            # ---- 规则 4：软上限 pre-flush ----
            if current_tokens + block_tokens > config.parent_target_tokens and current_texts:
                flush()

            # ---- 累积当前 block ----
            current_texts.append(text)
            current_blocks.append(block)
            current_tokens += block_tokens

            # ---- 规则 5：硬上限 flush ----
            if current_tokens >= config.parent_max_tokens:
                flush()

        # 循环结束，flush 剩余内容
        flush()
        return parents

    # ==================================================================
    # 阶段二：子块切分 (_split_into_children)
    # ==================================================================
    # 对每个父块内部按 block 类型切成 ~128 token 的子块。
    #
    # 切分规则：
    #   1. heading      → 单块一个子块（标题召回价值高）
    #   2. code/table   → 整体一个子块（不拆，语义完整）
    #   3. paragraph等  → 分三种情况：
    #      a) 段落 token ≤ child_target → 整段一个子块
    #      b) 段落 token > child_target  → 按句子切 + 贪心打包
    #      c) 无法切分（无标点长串）    → 整体保留（退化）
    #
    # 特殊父块（超长单 block / preserve 单 block）：
    #   父块只有一个 block → 直接对其文本做句子级切分（或整体保留 code/table）。
    # ==================================================================

    def _split_into_children(self, parent: dict, config: ChunkConfig) -> list[dict]:
        """把一个父块内部切成子块。

        参数：
            parent: 父块 dict，包含 text、source_blocks、trace
            config: 分块配置

        返回：
            子块列表，每个子块 dict 包含 text、source_blocks、trace
        """
        source_blocks = list(parent.get("source_blocks", []))
        if not source_blocks:
            return []

        children: list[dict] = []
        parent_trace = dict(parent.get("trace", {}))

        # ── 分支 A：父块只有单个 block ──
        #   可能是超长段落（split_long_block）或 preserve 的 code/table。
        #   直接对父块文本做切分，不需要逐 block 遍历。
        if parent_trace.get("split_long_block") or len(source_blocks) == 1:
            only_block = source_blocks[0]
            block_type = self._block_type(only_block)
            text = str(parent.get("text", "")).strip()
            if not text:
                return []

            # preserve 的 code/table 不拆，作为单个子块
            if self._preserve_whole(only_block, config) and block_type in {"code", "table"}:
                children.append(
                    self._make_child(
                        text=text,
                        source_blocks=[only_block],
                        split_by=block_type,
                    )
                )
                return children

            # 按句子拆 + 贪心打包
            fragments = self._split_by_sentence(text)
            # 只有一个片段 → 整体保留（防止无标点长串丢失内容）
            if len(fragments) <= 1:
                children.append(
                    self._make_child(
                        text=text,
                        source_blocks=[only_block],
                        split_by="paragraph",
                    )
                )
                return children
            # 多个片段 → 贪心打包
            for fragment in self._greedy_pack_sentences(
                fragments, config.child_target_tokens, config.child_max_tokens
            ):
                children.append(
                    self._make_child(
                        text=fragment,
                        source_blocks=[only_block],
                        split_by="sentence",
                    )
                )
            return children

        # ── 分支 B：正常父块（多个 source_blocks）──
        #   逐 block 遍历，根据类型分别处理。
        for block in source_blocks:
            block_type = self._block_type(block)
            text = str(self._read(block, "text", "")).strip()
            if not text:
                continue

            # heading：标题单独成子块（召回价值最高）
            if block_type == "heading":
                children.append(
                    self._make_child(text=text, source_blocks=[block], split_by="heading")
                )
                continue

            # code/table：整体成子块（不拆散语义）
            if self._preserve_whole(block, config) and block_type in {"code", "table"}:
                children.append(
                    self._make_child(text=text, source_blocks=[block], split_by=block_type)
                )
                continue

            # 普通段落：token 数判决
            block_tokens = max(1, self._counter.count(text))

            # 短段 → 整段一个子块
            if block_tokens <= config.child_target_tokens:
                children.append(
                    self._make_child(text=text, source_blocks=[block], split_by="paragraph")
                )
                continue

            # 长段 → 按句子切分 + 贪心打包
            fragments = self._split_by_sentence(text)
            if len(fragments) <= 1:
                # 没有句子边界（如无标点长串）：退化为整体，不丢内容
                children.append(
                    self._make_child(text=text, source_blocks=[block], split_by="paragraph")
                )
                continue
            for fragment in self._greedy_pack_sentences(
                fragments, config.child_target_tokens, config.child_max_tokens
            ):
                children.append(
                    self._make_child(text=fragment, source_blocks=[block], split_by="sentence")
                )

        # 兜底：所有 block 都为空时保留父块文本（理论上不应走到此分支）
        if not children:
            children.append(
                self._make_child(
                    text=str(parent.get("text", "")).strip(),
                    source_blocks=source_blocks,
                    split_by="paragraph",
                )
            )
        return children

    def _make_child(self, text: str, source_blocks: list, split_by: str) -> dict:
        """创建子块 dict 的工厂方法。

        参数：
            text: 子块文本内容
            source_blocks: 子块来源的原始 block 列表
            split_by: 切分方式标记（heading/paragraph/sentence/code/table）
        """
        return {
            "text": text,
            "source_blocks": source_blocks,
            "trace": {"split_by": split_by, "chunk_role": "child"},
        }

    # ==================================================================
    # 主入口：merge()
    # ==================================================================

    def merge(self, blocks: list, config: ChunkConfig) -> list[dict]:
        """两阶段切分：先父后子。

        流程：
            1. _merge_into_parents() → 父块列表
            2. 对每个父块，先写出父块自身到结果列表
            3. 对每个父块，_split_into_children() → 子块列表追加到结果列表
            4. 返回扁平的 parent+child 混合列表（深度优先，父块后紧跟其子块）

        每项 dict 结构：
            parent:
                {text, source_blocks, trace: {chunk_role: "parent", ...},
                 chunk_role: "parent"}

            child:
                {text, source_blocks, trace: {chunk_role: "child", split_by: ...},
                 chunk_role: "child", parent_ref: <parent_item>}

        注：parent_ref 是临时引用，由 MarkdownChunker 在分配 chunk_id 后回填
        parent_id/child_ids。

        参数：
            blocks: parsing 输出的结构化 block 列表
            config: 分块配置

        返回：
            扁平化的 parent+child 混合列表，按文档顺序深度优先排列
        """
        # 阶段一：合并成父块
        parents = self._merge_into_parents(blocks, config)

        merged: list[dict] = []
        for parent in parents:
            # 写出父块：标记 chunk_role="parent"
            parent_item = dict(parent)
            parent_item["chunk_role"] = "parent"
            trace = dict(parent_item.get("trace", {}))
            trace["chunk_role"] = "parent"
            parent_item["trace"] = trace
            merged.append(parent_item)

            # 阶段二：切分子块，每个子块带 parent_ref 临时引用
            children = self._split_into_children(parent, config)
            for child in children:
                child_item = dict(child)
                child_item["chunk_role"] = "child"
                # 临时引用：MarkdownChunker 据此回填 parent_id/child_ids
                child_item["parent_ref"] = parent_item
                merged.append(child_item)

        return merged