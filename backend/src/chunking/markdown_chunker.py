"""Markdown 文档分块器：将 parsing 阶段的 block 列表切分为 parent/child chunk。

================================================================================
工作流（三遍扫描）
================================================================================

输入：parsing 阶段的 ParseResultBlock 列表
      ↓
 MarkdownChunker.chunk(parse_blocks, chunk_config)
      ↓
 第一遍：调 BlockMergeStrategy.merge() 完成两阶段切分 → parent + child 混合列表
 第二遍：为每个 item 分配全局 chunk_id，建立 parent_chunk_id_by_item 映射表
 第三遍：回填父子关系（parent_id / child_ids），组装 ChunkRecord + ChunkMeta 输出
      ↓
输出：list[dict]（标准化的 chunk 记录，每个 dict 包含 chunk_id, text, meta 等）

================================================================================
父子分块 Small-to-Big 检索策略
================================================================================

检索时的工作流程：
  用户查询 "什么是自旋锁"
       ↓ 向量相似度搜索
  命中 Child "自旋锁（无标准库实现..."（~128 tokens，精准匹配）
       ↓ child.parent_id
  回溯 Parent（~512 tokens，包含完整章节上下文）
       ↓ 返回
  最终返回：Parent 的完整文本（含标题、自旋锁对比表、代码示例）

================================================================================
输出格式
================================================================================

每条 chunk 输出的 dict 结构：

  {
    "chunk_id":     "doc_001_ck_12",          # 全局唯一 chunk ID
    "doc_id":       "doc_001",                # 所属文档 ID
    "text":         "...",                     # chunk 文本内容
    "content":      "...",                     # text 的副本（兼容旧接口）
    "section_path": ["二、锁", "2.1 mutex"],  # 章节路径
    "page_no":      None,                      # 页码（Markdown 为 None）
    "chunk_role":   "parent" | "child",       # 分块角色
    "parent_id":    "doc_001_ck_12" | None,   # 父块 ID（子块有值）
    "child_ids":    ["doc_001_ck_13", ...],   # 子块 ID 列表（父块有值）
    "chunk_order":  12,                        # 全局顺序号
    "meta":         { ... },                   # 完整元数据（ChunkMeta 序列化）
  }
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from backend.src.chunking.block_merge import BlockMergeStrategy
from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.models import ChunkMeta, ChunkRecord, validate_chunk_record
from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.parsing.models import SourceSpan, coerce_source_span


class MarkdownChunker:
    """Markdown 文档分块器。

    将 parsing 阶段产出的 ParseResultBlock 列表切分为父子双粒度的 chunk 列表。
    内部三遍扫描，确保 chunk_id、parent_id、child_ids 全部正确关联。

    使用示例：
        >>> chunker = MarkdownChunker()
        >>> parse_blocks = [...]  # parsing 模块的输出
        >>> config = ChunkConfig(parent_target_tokens=512, child_target_tokens=128)
        >>> chunks = chunker.chunk(parse_blocks, config)
        >>> for c in chunks:
        ...     print(c["chunk_id"], c["chunk_role"], c["text"][:50])
    """

    def __init__(self) -> None:
        """初始化分块器：创建合并策略和 token 计数器实例。"""
        self._merge = BlockMergeStrategy()
        self._counter = SimpleTokenCounter()

    # ==================================================================
    # 基础工具方法
    # ==================================================================

    def _read(self, block, key: str, default=None):
        """安全读取 block 字段（兼容 dict / 对象），统一使用此方法访问 block 属性。"""
        if isinstance(block, dict):
            return block.get(key, default)
        return getattr(block, key, default)

    def _unique(self, values: list[str]) -> list[str]:
        """保持顺序的去重工具（用于 source_block_ids 等可能重复的字段）。"""
        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    # ==================================================================
    # SourceSpan 合并工具
    # ==================================================================

    def _merge_span(self, source_blocks: list) -> SourceSpan | None:
        """合并多个 block 的 source_span 为一个整体 span。

        规则：
        - start_line / start_char 取第一个有值的 block 的值（最小边界）。
        - end_line / end_char   取最后一个有值的 block 的值（最大边界）。
        - page_no               取第一个有值的 block 的页码。
        """
        spans = [coerce_source_span(self._read(block, "source_span")) for block in source_blocks]
        spans = [span for span in spans if span is not None]
        if not spans:
            return None
        start_line = next((s.start_line for s in spans if s.start_line is not None), None)
        end_line = next((s.end_line for s in reversed(spans) if s.end_line is not None), None)
        start_char = next((s.start_char for s in spans if s.start_char is not None), None)
        end_char = next((s.end_char for s in reversed(spans) if s.end_char is not None), None)
        page_no = next((s.page_no for s in spans if s.page_no is not None), None)
        return SourceSpan(
            start_line=start_line,
            end_line=end_line,
            start_char=start_char,
            end_char=end_char,
            page_no=page_no,
            accuracy=(
                "exact"
                if spans and all(s.accuracy == "exact" for s in spans)
                else "line_only"
                if any(s.accuracy in {"exact", "line_only"} for s in spans)
                else "unavailable"
            ),
        )

    def _child_span(self, item: dict) -> SourceSpan | None:
        """Project a normalized Parent range back to raw source when possible."""
        start = item.get("parent_char_start")
        end = item.get("parent_char_end")
        if start is None or end is None:
            return self._merge_span(list(item.get("source_blocks", [])))
        projected: list[SourceSpan] = []
        for segment in item.get("parent_segment_map", []):
            overlap_start = max(int(start), int(segment["parent_start"]))
            overlap_end = min(int(end), int(segment["parent_end"]))
            if overlap_start >= overlap_end:
                continue
            block_span = coerce_source_span(self._read(segment["block"], "source_span"))
            if block_span is None:
                continue
            raw_start = block_span.start_char
            raw_end = block_span.end_char
            accuracy = block_span.accuracy
            if accuracy == "exact" and raw_start is not None:
                raw_start = (
                    raw_start
                    + int(segment["block_char_start"])
                    + overlap_start
                    - int(segment["parent_start"])
                )
                raw_end = raw_start + (overlap_end - overlap_start)
            else:
                accuracy = "line_only" if accuracy != "unavailable" else "unavailable"
            projected.append(
                SourceSpan(
                    start_line=block_span.start_line,
                    end_line=block_span.end_line,
                    start_char=raw_start,
                    end_char=raw_end,
                    page_no=block_span.page_no,
                    accuracy=accuracy,
                )
            )
        if not projected:
            return self._merge_span(list(item.get("source_blocks", [])))
        return self._merge_span([{"source_span": span} for span in projected])

    def _span_dict(self, span: SourceSpan | None) -> dict | None:
        """将 SourceSpan 对象序列化为 dict（None-safe）。"""
        if span is None:
            return None
        return {
            "start_line": span.start_line,
            "end_line": span.end_line,
            "start_char": span.start_char,
            "end_char": span.end_char,
            "page_no": span.page_no,
            "accuracy": span.accuracy,
        }

    # ==================================================================
    # 主入口：chunk()
    # ==================================================================

    def chunk(self, parse_blocks: list, chunk_config: ChunkConfig | dict) -> list[dict]:
        """对 parsing 产出执行父子双粒度分块。

        三遍扫描：
            第一遍：调 BlockMergeStrategy.merge() → parent + child 混合列表，
                   分配 chunk_id，建立 parent 映射表。
            第二遍：回填子块的 parent_id。
            第三遍：生成完整输出，组装 ChunkRecord + ChunkMeta，校验并序列化。

        参数：
            parse_blocks: parsing 阶段产出的 block 列表（dict 或 ParseResultBlock）
            chunk_config:  分块配置（ChunkConfig 实例或 dict）

        返回：
            list[dict]：扁平化的 chunk 记录，按文档出现顺序排列，包含完整的
            chunk_id、text、meta、父子关系信息。
        """
        # ── 配置解析 ──
        config = build_chunk_config(chunk_config)
        profile_hash = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]

        # ── 阶段一：两阶段切分（父块合并 + 子块拆分）──
        merged = self._merge.merge(parse_blocks, config)

        # ── 确定 doc_id ──
        # 从第一个有文本的 block 获取 doc_id
        doc_id = "doc_unknown"
        for block in parse_blocks:
            text = str(self._read(block, "text", "")).strip()
            if text:
                doc_id = str(self._read(block, "doc_id", "doc_unknown"))
                break

        # ══════════════════════════════════════════════════════════════
        # 第一遍扫描：分配全局 chunk_id，建立 parent 映射表
        # ══════════════════════════════════════════════════════════════
        # items_with_id: [(order, item, chunk_id), ...]
        # parent_chunk_id_by_item: {id(parent_item): chunk_id}
        #   用 id(item) 作为 key 而不是 item 自身的某个字段，
        #   因为同一个 parent_item 对象在 merged 列表中只出现一次（对象相同）。
        # ══════════════════════════════════════════════════════════════

        items_with_id: list[tuple[int, dict, str]] = []  # (order, item, chunk_id)
        parent_chunk_id_by_item: dict[int, str] = {}       # id(item) → chunk_id

        order = 0
        for item in merged:
            order += 1
            chunk_id = f"{doc_id}_v2_{profile_hash}_{order}"
            items_with_id.append((order, item, chunk_id))
            # 父块记入映射表，供子块回填 parent_id
            if item.get("chunk_role") == "parent":
                parent_chunk_id_by_item[id(item)] = chunk_id

        # ══════════════════════════════════════════════════════════════
        # 第二遍扫描：回填子块的 parent_id
        # ══════════════════════════════════════════════════════════════
        # 子块有 parent_ref 临时引用（指向父块 item），通过 id(parent_ref) 查映射表
        # 获取父块的 chunk_id，回填到子块的 parent_id 字段。
        # ══════════════════════════════════════════════════════════════

        for order, item, chunk_id in items_with_id:
            if item.get("chunk_role") == "child":
                parent_ref = item.get("parent_ref")
                parent_id = parent_chunk_id_by_item.get(id(parent_ref)) if parent_ref else None
                item["parent_id"] = parent_id

        # ── 构建 child_ids_by_parent_id 反向索引 ──
        child_ids_by_parent_id: dict[str, list[str]] = {}
        for order, item, chunk_id in items_with_id:
            if item.get("chunk_role") == "child":
                parent_id = item.get("parent_id")
                if parent_id:
                    child_ids_by_parent_id.setdefault(parent_id, []).append(chunk_id)

        # ══════════════════════════════════════════════════════════════
        # 第三遍扫描：生成输出，组装 ChunkRecord + ChunkMeta
        # ══════════════════════════════════════════════════════════════

        rows: list[dict] = []
        for order, item, chunk_id in items_with_id:
            # ── 获取 source_blocks（当前 chunk 的原始 block 列表）──
            source_blocks = list(item.get("source_blocks", []))
            if not source_blocks:
                continue

            # ── 取第一个 block 作为"代表 block"（用于获取 doc_id/section_path）──
            first = source_blocks[0]

            # ── chunk 基本信息 ──
            chunk_doc_id = str(self._read(first, "doc_id", doc_id))
            text = str(item.get("text", ""))
            if not text.strip():
                continue

            # ── 合并 source_span ──
            source_span = (
                self._child_span(item)
                if item.get("chunk_role") == "child"
                else self._merge_span(source_blocks)
            )

            # ── chunk_role 和父子 ID ──
            chunk_role = str(item.get("chunk_role", "parent"))
            parent_id = item.get("parent_id")
            child_ids = (
                list(child_ids_by_parent_id.get(chunk_id, [])) if chunk_role == "parent" else []
            )

            # ── trace 信息（切分追踪，清理内部字段后保留）──
            base_trace = dict(item.get("trace", {}))
            base_trace.pop("chunk_role", None)  # 清理临时引用，不进输出

            # ── 构建 ChunkMeta 元数据 ──
            meta = ChunkMeta(
                # 章节路径（来自第一个 source_block）
                section_path=list(self._read(first, "section_path", [])),
                field_path=[],
                chunk_order=order,
                mom_id=None,
                # 组成此 chunk 的所有 block 类型
                block_types=[
                    str(self._read(b, "block_type", "paragraph")) for b in source_blocks
                ],
                # 去重的 source_block_ids
                source_block_ids=self._unique(
                    [
                        f"{chunk_doc_id}_blk_{int(self._read(b, 'order', order))}"
                        for b in source_blocks
                    ]
                ),
                # 近似 token 数
                token_count=self._counter.count(text),
                # 页码信息
                page_no=self._read(first, "page_no"),
                # 合并后的溯源跨度
                source_span=source_span,
                # 切分追踪
                trace=base_trace,
                # 父子关系
                chunk_role=chunk_role,
                parent_id=parent_id,
                child_ids=child_ids,
                retrieval_eligible=chunk_role == "child",
                parent_char_start=item.get("parent_char_start"),
                parent_char_end=item.get("parent_char_end"),
            )

            # ── 构建 ChunkRecord 并校验 ──
            record = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=chunk_doc_id,
                text=text,
                meta=meta,
            )
            # 防御式校验：空 chunk_id / doc_id / text 会抛 ValueError
            validate_chunk_record(record)

            # ── 序列化为 dict 输出 ──
            rows.append(
                {
                    # ---- 顶层字段（下游检索直接读取）----
                    "chunk_id": record.chunk_id,
                    "doc_id": record.doc_id,
                    "text": record.text,
                    "content": record.text,  # 兼容旧接口
                    "section_path": record.meta.section_path,
                    "page_no": record.meta.page_no,
                    # ---- 父子分块关键字段 ----
                    "chunk_role": record.meta.chunk_role,
                    "parent_id": record.meta.parent_id,
                    "child_ids": list(record.meta.child_ids),
                    "chunk_order": record.meta.chunk_order,
                    "chunk_profile_version": "parent-child-v2",
                    "chunk_profile_hash": profile_hash,
                    "retrieval_eligible": record.meta.retrieval_eligible,
                    "source_span": self._span_dict(record.meta.source_span),
                    "source_block_ids": list(record.meta.source_block_ids),
                    "parent_char_start": record.meta.parent_char_start,
                    "parent_char_end": record.meta.parent_char_end,
                    # ---- 完整元数据（序列化为嵌套 dict）----
                    "meta": {
                        "field_path": record.meta.field_path,
                        "chunk_order": record.meta.chunk_order,
                        "mom_id": record.meta.mom_id,
                        "block_types": record.meta.block_types,
                        "source_block_ids": record.meta.source_block_ids,
                        "token_count": record.meta.token_count,
                        "page_no": record.meta.page_no,
                        "source_span": self._span_dict(record.meta.source_span),
                        "trace": record.meta.trace,
                        "chunk_role": record.meta.chunk_role,
                        "parent_id": record.meta.parent_id,
                        "child_ids": list(record.meta.child_ids),
                        "retrieval_eligible": record.meta.retrieval_eligible,
                        "parent_char_start": record.meta.parent_char_start,
                        "parent_char_end": record.meta.parent_char_end,
                    },
                }
            )

        return rows
