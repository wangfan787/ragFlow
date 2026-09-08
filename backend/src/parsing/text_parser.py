from __future__ import annotations

import re

from langchain_core.documents import Document

from .models import ParseSource, parse_source_from_config, parsed_documents


class TextParser:
    """Parse plain text into paragraph blocks while retaining raw offsets."""

    _PARAGRAPH_RE = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)

    def parse(self, doc_id: str, parse_config: dict | ParseSource) -> list[Document]:
        source = (
            parse_config if isinstance(parse_config, ParseSource) else parse_source_from_config(parse_config)
        )
        text = source.read_text()
        line_starts = [0]
        line_starts.extend(match.end() for match in re.finditer("\n", text))

        def line_no(offset: int) -> int:
            import bisect

            return bisect.bisect_right(line_starts, offset)

        blocks: list[dict] = []
        for order, match in enumerate(self._PARAGRAPH_RE.finditer(text), start=1):
            content = match.group(0)
            blocks.append(
                {
                    "text": content,
                    "block_type": "paragraph",
                    "page_no": None,
                    "bbox": None,
                    "section_path": [],
                    "order": order,
                    "source_span": {
                        "start_line": line_no(match.start()),
                        "end_line": line_no(max(match.start(), match.end() - 1)),
                        "start_char": match.start(),
                        "end_char": match.end(),
                        "page_no": None,
                        "accuracy": "exact",
                    },
                    "metadata": {"source_type": source.source_type or "text"},
                }
            )
        if not blocks:
            raise ValueError("no parseable content found")
        return parsed_documents(blocks, doc_id, source.name)
