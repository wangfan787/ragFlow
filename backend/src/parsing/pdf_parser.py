# 导入未来特性以支持延迟注解求值
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from langchain_core.documents import Document

from .models import ParseSource, parse_source_from_config, parsed_documents


class PdfParser:
    """PDF 解析器，基于 pypdf 库提取文本内容。"""

    def parse(self, doc_id: str, parse_config: dict | ParseSource) -> list[Document]:
        """解析 PDF 文件，按段落输出结构化 block 列表。"""
        source = (
            parse_config if isinstance(parse_config, ParseSource) else parse_source_from_config(parse_config)
        )
        file_path = source.file_path or Path("")
        if not file_path.exists():
            raise FileNotFoundError(f"source file missing for doc {doc_id}: {file_path}")

        reader = PdfReader(str(file_path))
        blocks: list[dict] = []
        order = 0

        # 逐页提取文本
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            # 按换行分割，过滤空行，每行作为一个段落块
            for para in [line.strip() for line in text.split("\n") if line.strip()]:
                order += 1
                blocks.append(
                    {
                        "text": para,
                        "block_type": "paragraph",
                        "page_no": page_no,
                        "bbox": None,
                        "section_path": [f"page-{page_no}"],   # 以页码作为章节路径
                        "order": order,
                        "source_span": {
                            "start_line": None,
                            "end_line": None,
                            "start_char": None,
                            "end_char": None,
                            "page_no": page_no,
                            "accuracy": "unavailable",
                        },
                        "metadata": {},
                    }
                )

        if not blocks:
            raise ValueError("no parseable content found")
        return parsed_documents(blocks, doc_id, source.name)
