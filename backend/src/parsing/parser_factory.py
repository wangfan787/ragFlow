# 导入未来特性以支持延迟注解求值
from __future__ import annotations

# 各格式解析器；全部输出统一的 ParseResultBlock
from .markdown_parser_it import MarkdownParserIt  # Markdown：基于 markdown-it-py
from .pdf_parser import PdfParser
from .text_parser import TextParser
from .html_parser import HtmlParser


def build_parser(file_type: str):
    """根据文件类型构建对应的解析器实例。

    支持：
    - md / markdown: Markdown（markdown-it-py）
    - pdf: PDF
    - txt / text: 纯文本
    - html / htm: HTML
    """
    normalized = (file_type or "").strip().lower()
    normalized = normalized.lstrip(".")
    if normalized in {"md", "markdown"}:
        return MarkdownParserIt()
    if normalized == "pdf":
        return PdfParser()
    if normalized in {"txt", "text"}:
        return TextParser()
    if normalized in {"html", "htm"}:
        return HtmlParser()
    raise ValueError(f"unsupported file_type for parser: {file_type}")
