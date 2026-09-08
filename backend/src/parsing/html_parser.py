from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

from langchain_core.documents import Document

from .models import ParseSource, parse_source_from_config, parsed_documents


class HtmlParser:
    """Parse HTML structure while explicitly degrading normalized raw spans."""

    _BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table")

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    def parse(self, doc_id: str, parse_config: dict | ParseSource) -> list[Document]:
        source = (
            parse_config if isinstance(parse_config, ParseSource) else parse_source_from_config(parse_config)
        )
        raw = source.read_text()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        descendants = list(soup.descendants)
        order_by_id = {id(node): index for index, node in enumerate(descendants)}
        events: list[tuple[int, str, str, str]] = []
        for element in soup.find_all(self._BLOCK_TAGS):
            if element.find_parent(self._BLOCK_TAGS):
                continue
            text = self._clean(
                element.get_text("\n" if element.name in {"pre", "table"} else " ", strip=True)
            )
            if not text:
                continue
            block_type = (
                "heading"
                if element.name.startswith("h")
                else "code"
                if element.name == "pre"
                else "table"
                if element.name == "table"
                else "list"
                if element.name == "li"
                else "paragraph"
            )
            fragment = str(element)
            events.append(
                (
                    order_by_id[id(element)],
                    text,
                    block_type,
                    fragment,
                )
            )

        # Preserve visible text outside recognized block elements. This is
        # common in HTML fragments using only <div>/<span>/<br> wrappers.
        for node in descendants:
            if not isinstance(node, NavigableString):
                continue
            parent = node.parent
            if parent is None or parent.name in {"script", "style", "noscript"}:
                continue
            if any(isinstance(ancestor, Tag) and ancestor.name in self._BLOCK_TAGS for ancestor in node.parents):
                continue
            text = self._clean(str(node))
            if not text:
                continue
            events.append(
                (
                    order_by_id[id(node)],
                    text,
                    "paragraph",
                    str(node),
                )
            )

        events.sort(key=lambda item: item[0])
        blocks: list[dict] = []
        section_path: list[str] = []
        normalized_cursor = 0
        raw_cursor = 0
        for _, text, block_type, raw_fragment in events:
            located = raw.find(raw_fragment, raw_cursor)
            raw_start = located if located >= 0 else None
            raw_end = located + len(raw_fragment) if located >= 0 else None
            if raw_end is not None:
                raw_cursor = raw_end
            if block_type == "heading":
                # Recover heading level from the raw tag when available.
                match = re.match(r"<h([1-6])\b", raw[raw_start:] if raw_start is not None else "", re.I)
                level = int(match.group(1)) if match else 1
                section_path = section_path[: max(0, level - 1)]
                section_path.append(text)
            accuracy = "line_only" if raw_start is not None else "unavailable"
            normalized_start = normalized_cursor
            normalized_cursor += len(text)
            blocks.append(
                {
                    "text": text,
                    "block_type": block_type,
                    "page_no": None,
                    "bbox": None,
                    "section_path": list(section_path),
                    "order": len(blocks) + 1,
                    "source_span": {
                        "start_line": raw.count("\n", 0, raw_start) + 1 if raw_start is not None else None,
                        "end_line": raw.count("\n", 0, raw_end) + 1 if raw_end is not None else None,
                        "start_char": raw_start,
                        "end_char": raw_end,
                        "page_no": None,
                        "accuracy": accuracy,
                    },
                    "metadata": {
                        "source_type": "html",
                        "normalization_segments": [
                            {
                                "normalized_start": normalized_start,
                                "normalized_end": normalized_cursor,
                                "raw_start": raw_start,
                                "raw_end": raw_end,
                                "accuracy": accuracy,
                            }
                        ],
                    },
                }
            )
            normalized_cursor += 2
        if not blocks:
            raise ValueError("no parseable content found")
        return parsed_documents(blocks, doc_id, source.name)
