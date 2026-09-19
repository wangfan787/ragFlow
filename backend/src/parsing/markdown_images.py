"""Markdown 本地图片引用提取与 image block 交错：纯文本操作，无 IO 副作用。

P0-A（docs/plan/算法层QA.md §7.2）只支持 Markdown 相对路径本地
PNG/JPEG/WebP。URL、绝对路径、`..` 穿越与不支持的扩展名一律产出
带原因的 skipped 引用，由调用方如实记录，不在本层静默丢弃。

source_span 直接取图片语法在原文中的正则匹配区间，精度 exact；
与 UnstructuredParser 文本块坐标同处一个坐标系，可按 start_char
交错合并。已知取舍：坐标一致性在原文含 HTML 实体的极端情况下可能与
文本块（基于实体解码后的原文回定位）有偏移，只影响交错位置，不影响
span 本身的诚实性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_core.documents import Document

# 只允许相对路径引用的本地位图；SVG 等矢量/文本格式不在首版范围。
SUPPORTED_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# ![alt](path) / ![alt](<path with space>) / ![alt](path "title")
_IMAGE_RE = re.compile(
    r"!\[([^\]\n]*)\]\(\s*(?:<([^>\n]*)>|([^)\s]+))(?:\s+\"[^\"]*\")?\s*\)"
)

# 围栏代码块中的图片语法是代码示例，不是真实引用。
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(```+|~~~+)")


@dataclass(frozen=True)
class ImageRef:
    """一次 Markdown 图片引用；candidate 带 resolved_path，否则带 skip_reason。"""

    alt: str
    raw_path: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    resolved_path: Path | None = None
    mime_type: str | None = None
    skip_reason: str | None = None


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    fence_marker: str | None = None
    fence_start = 0
    cursor = 0
    for line in text.splitlines(keepends=True):
        if fence_marker is None:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                fence_marker = match.group(1)
                fence_start = cursor
        elif line.strip().startswith(fence_marker):
            ranges.append((fence_start, cursor + len(line)))
            fence_marker = None
        cursor += len(line)
    if fence_marker is not None:  # 未闭合围栏按到文末处理
        ranges.append((fence_start, len(text)))
    return ranges


def _skip_reason(raw_path: str, base_dir: Path) -> str | None:
    path = raw_path.strip()
    if not path:
        return "empty_path"
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", path) or path.startswith(("data:", "mailto:")):
        return "external_url"
    if path.startswith("#"):
        return "anchor"
    candidate = Path(path)
    if candidate.is_absolute():
        return "absolute_path"
    if ".." in candidate.parts:
        return "path_traversal"
    if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        return "unsupported_type"
    if not (base_dir / candidate).resolve().is_relative_to(base_dir.resolve()):
        return "path_traversal"
    if not (base_dir / candidate).is_file():
        return "file_not_found"
    return None


def extract_image_refs(markdown_text: str, base_dir: Path) -> list[ImageRef]:
    """扫描全文图片语法；文件存在性检查是唯一的只读 IO。"""
    fenced = _fenced_ranges(markdown_text)
    refs: list[ImageRef] = []
    for match in _IMAGE_RE.finditer(markdown_text):
        if any(start <= match.start() < end for start, end in fenced):
            continue
        alt = (match.group(1) or "").strip()
        raw_path = (match.group(2) if match.group(2) is not None else match.group(3) or "").strip()
        start_char, end_char = match.start(), match.end()
        ref = ImageRef(
            alt=alt,
            raw_path=raw_path,
            start_char=start_char,
            end_char=end_char,
            start_line=markdown_text.count("\n", 0, start_char) + 1,
            end_line=markdown_text.count("\n", 0, end_char) + 1,
        )
        reason = _skip_reason(raw_path, base_dir)
        if reason is not None:
            refs.append(replace(ref, skip_reason=reason))
        else:
            relative = Path(raw_path)
            refs.append(
                replace(
                    ref,
                    resolved_path=(base_dir / relative).resolve(),
                    mime_type=SUPPORTED_IMAGE_EXTS[relative.suffix.lower()],
                )
            )
    return refs


def _span_range(block: Document) -> tuple[float, float] | None:
    span = block.metadata.get("source_span") or {}
    start, end = span.get("start_char"), span.get("end_char")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return float(start), float(end)
    return None


def _block_start_char(block: Document) -> float:
    span = _span_range(block)
    return span[0] if span is not None else float("inf")


def interleave_image_blocks(
    text_blocks: list[Document], image_blocks: list[Document],
) -> list[Document]:
    """按 source_span 把 image block 插入文本块流，并重排 order。

    图片语法中的 alt 文本若被引擎单独输出为段落（span 完全落在图片
    区间内），视为同一内容的重复，直接丢弃。image block 的
    section_path 继承其前方最近文本块，保证章节上下文可用。
    """
    image_spans = [
        (block.metadata["source_span"]["start_char"], block.metadata["source_span"]["end_char"])
        for block in image_blocks
    ]
    kept = []
    for block in text_blocks:
        span = _span_range(block)
        inside_image = span is not None and any(
            start <= span[0] and span[1] <= end for start, end in image_spans
        )
        if not inside_image:
            kept.append(block)

    pending = sorted(image_blocks, key=lambda block: _block_start_char(block))
    merged: list[Document] = []
    section_path: list[str] = []
    for block in kept:
        start = _block_start_char(block)
        while pending and _block_start_char(pending[0]) < start:
            image = pending.pop(0)
            image.metadata["section_path"] = list(section_path)
            merged.append(image)
        merged.append(block)
        if block.metadata.get("block_type") == "heading":
            section_path = list(block.metadata.get("section_path", []))
    for image in pending:
        image.metadata["section_path"] = list(section_path)
        merged.append(image)

    for order, block in enumerate(merged, start=1):
        block.metadata["order"] = order
    return merged
