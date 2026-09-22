"""提取 Markdown 页头，保护代码和表格的原文与来源。"""

import re

from markdown_it import MarkdownIt


_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def prepare_markdown(raw: str) -> tuple[str, dict[str, dict], dict | None]:
    """结构块替换为独立占位段落，普通内容仍交给通用解析器。"""
    frontmatter = None
    match = _FRONTMATTER_RE.match(raw)
    if match:
        frontmatter = {
            "text": match.group(1), "block_type": "frontmatter",
            "page_no": None, "bbox": None, "section_path": [], "order": 0,
            "source_span": {
                "start_line": 1, "end_line": len(match.group().splitlines()),
                "start_char": 0, "end_char": match.end(),
                "page_no": None, "accuracy": "line_only",
            },
        }
        # 留下空白以保持后续结构块在原文中的行号和字符偏移。
        raw = re.sub(r"[^\r\n]", " ", raw[:match.end()]) + raw[match.end():]
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = MarkdownIt("commonmark").enable("table").parse(raw)
    prefix = "ragflowprotectedstructure"
    while prefix in raw:
        prefix += "x"
    protected: dict[str, dict] = {}
    parts: list[str] = []
    cursor = 0
    for token in tokens:
        if token.type not in {"fence", "code_block", "table_open"} or token.map is None:
            continue
        start_line, end_line = token.map
        start, end = offsets[start_line], offsets[end_line]
        marker = f"{prefix}{len(protected)}"
        protected[marker] = {
            "text": raw[start:end],
            "block_type": "table" if token.type == "table_open" else "code",
            "source_span": {
                "start_line": start_line + 1, "end_line": end_line,
                "start_char": start, "end_char": end,
                "page_no": None, "accuracy": "exact",
            },
        }
        parts.extend([raw[cursor:start], f"\n\n{marker}\n\n"])
        cursor = end
    parts.append(raw[cursor:])
    return "".join(parts), protected, frontmatter
