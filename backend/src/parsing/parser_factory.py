# 导入未来特性以支持延迟注解求值
from __future__ import annotations

# 导入 Markdown 和 PDF 解析器
from .markdown_parser_it import MarkdownParserIt  # 使用基于 markdown-it-py 的解析器
from .pdf_parser import PdfParser


def build_parser(file_type: str):
    """根据文件类型构建对应的解析器实例。

    支持：
    - md: Markdown 文件（使用 markdown-it-py，标准兼容）
    - pdf: PDF 文件
    """
    normalized = (file_type or "").strip().lower()
    if normalized == "md":
        return MarkdownParserIt()
    if normalized == "pdf":
        return PdfParser()
    raise ValueError(f"unsupported file_type for parser: {file_type}")