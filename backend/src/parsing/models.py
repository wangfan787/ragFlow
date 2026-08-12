# 解析阶段的数据结构。
# SourceSpan 描述文本在原始文档中的位置（行号/字符/页码），
# 被解析、切分、检索阶段共用，所以定义在最早的 parsing 阶段。

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceSpan:
    start_line: int | None = None   # 起始行号
    end_line: int | None = None     # 结束行号
    start_char: int | None = None   # 起始字符位置
    end_char: int | None = None     # 结束字符位置
    page_no: int | None = None      # 页码
    accuracy: str = "unavailable"  # exact / line_only / unavailable


@dataclass(frozen=True)
class ParseSource:
    """A parser input that may come from a file or in-memory text."""

    source_type: str
    name: str = ""
    file_path: Path | None = None
    text: str | None = None

    def read_text(self) -> str:
        if self.text is not None:
            return self.text
        if self.file_path is None:
            raise ValueError("parse source requires text or file_path")
        if not self.file_path.exists():
            raise FileNotFoundError(f"source file missing: {self.file_path}")
        try:
            return self.file_path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"source is not valid UTF-8; provide an explicitly converted UTF-8 file: {self.file_path}"
            ) from exc


def parse_source_from_config(parse_config: dict) -> ParseSource:
    source_type = str(parse_config.get("file_type") or parse_config.get("source_type") or "")
    file_value = parse_config.get("file_path")
    return ParseSource(
        source_type=source_type.strip().lower().lstrip("."),
        name=str(parse_config.get("doc_name") or (Path(file_value).name if file_value else "")),
        file_path=Path(file_value) if file_value else None,
        text=parse_config.get("text"),
    )


def coerce_source_span(span) -> SourceSpan | None:
    # 若已是 SourceSpan 或 None 则直接返回，若为 dict 则转换
    if span is None or isinstance(span, SourceSpan):
        return span
    if isinstance(span, dict):
        return SourceSpan(
            start_line=span.get("start_line"),
            end_line=span.get("end_line"),
            start_char=span.get("start_char"),
            end_char=span.get("end_char"),
            page_no=span.get("page_no"),
            accuracy=str(span.get("accuracy", "unavailable")),
        )
    return None


@dataclass
class ParseResultBlock:
    """Parser 统一输出的结构化 block，是切分阶段的输入。"""

    text: str                       # 文本内容
    block_type: str                 # 块类型（如标题、段落等）
    page_no: int | None             # 所在页码
    bbox: list[float] | None        # 边界框坐标
    section_path: list[str]         # 章节路径
    order: int = 0                  # 顺序号
    source_span: SourceSpan | None = None   # 原始位置信息
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据
