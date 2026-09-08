"""解析输入与 Document 输出；来源信息直接放在 metadata 中。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass(frozen=True)
class ParseSource:
    """文件或内存文本，统一严格 UTF-8 读取。"""
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
            raise ValueError(f"source is not valid UTF-8: {self.file_path}") from exc


def parse_source_from_config(parse_config: dict) -> ParseSource:
    source_type = str(parse_config.get("file_type") or parse_config.get("source_type") or "")
    file_value = parse_config.get("file_path")
    return ParseSource(
        source_type=source_type.strip().lower().lstrip("."),
        name=str(parse_config.get("doc_name") or (Path(file_value).name if file_value else "")),
        file_path=Path(file_value) if file_value else None,
        text=parse_config.get("text"),
    )


def parsed_documents(blocks: list[dict], doc_id: str, name: str) -> list[Document]:
    """在解析边界封装一次；不把正文同时复制到 metadata。"""
    return [
        Document(
            page_content=block["text"],
            metadata={
                **block.get("metadata", {}),
                **{key: value for key, value in block.items() if key not in {"text", "metadata"}},
                "doc_id": doc_id,
                "doc_name": name,
            },
        )
        for block in blocks
    ]
