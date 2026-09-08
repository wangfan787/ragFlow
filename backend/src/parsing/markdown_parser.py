# 导入未来特性以支持延迟注解求值
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from .models import parsed_documents

# 正则：匹配 Markdown 标题（如 # 标题）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
# 正则：匹配无序列表（- * +）或有序列表（1.）
_LIST_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+).+")
# 正则：匹配水平分割线（---, ***, ___）
_HR_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")


class MarkdownParser:
    """Markdown 解析器，显式支持 GFM 表格。"""

    def _is_special(self, line: str) -> bool:
        """判断当前行是否为特殊块（标题、代码块、表格、列表、引用、分割线）。"""
        stripped = line.strip()
        if not stripped:
            return True
        return bool(
            _HEADING_RE.match(stripped)
            or stripped.startswith("```")
            or stripped.startswith("|")
            or _LIST_RE.match(stripped)
            or stripped.startswith(">")
            or _HR_RE.match(stripped)
        )

    def parse(self, doc_id: str, parse_config: dict) -> list[Document]:
        """解析 Markdown 文件，输出结构化 block 列表。"""
        file_path = Path(parse_config.get("file_path", ""))
        if not file_path.exists():
            raise FileNotFoundError(f"source file missing for doc {doc_id}: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        blocks: list[dict] = []
        order = 0
        section_path: list[str] = []   # 当前章节路径

        def add_block(
            block_type: str, block_text: str, start_line: int, end_line: int, section: list[str]
        ) -> None:
            """向 blocks 中添加一个结构化块。"""
            nonlocal order
            clean = block_text.strip()
            if not clean:
                return
            order += 1
            blocks.append(
                {
                    "text": clean,
                    "block_type": block_type,
                    "page_no": None,
                    "bbox": None,
                    "section_path": list(section),
                    "order": order,
                    "source_span": {
                        "start_line": start_line,
                        "end_line": end_line,
                        "start_char": 0,
                        "end_char": len(clean),
                        "page_no": None,
                    },
                    "metadata": {},
                }
            )

        idx = 0
        # 解析 frontmatter（YAML 元数据头，以 --- 包裹）
        if len(lines) >= 3 and lines[0].strip() == "---":
            end = 1
            while end < len(lines) and lines[end].strip() != "---":
                end += 1
            if end < len(lines):
                add_block("frontmatter", "\n".join(lines[1:end]), 1, end + 1, section_path)
                idx = end + 1

        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()
            if not stripped:
                idx += 1
                continue

            # 处理标题块
            heading = _HEADING_RE.match(stripped)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                # 根据标题级别截断章节路径
                section_path = section_path[: max(0, level - 1)]
                section_path.append(title)
                add_block("heading", title, idx + 1, idx + 1, section_path)
                idx += 1
                continue

            # 处理代码块（```）
            if stripped.startswith("```"):
                start = idx
                fence = stripped[:3]   # 获取 fence 标记
                idx += 1
                body: list[str] = []
                while idx < len(lines) and not lines[idx].strip().startswith(fence):
                    body.append(lines[idx])
                    idx += 1
                if idx < len(lines):
                    idx += 1   # 跳过结束 fence
                add_block("code", "\n".join(body), start + 1, idx, section_path)
                continue

            # 处理表格（以 | 开头）
            if stripped.startswith("|"):
                start = idx
                body = [raw]
                idx += 1
                while idx < len(lines) and lines[idx].strip().startswith("|"):
                    body.append(lines[idx])
                    idx += 1
                add_block("table", "\n".join(body), start + 1, idx, section_path)
                continue

            # 处理列表项
            if _LIST_RE.match(stripped):
                start = idx
                body = [raw]
                idx += 1
                while idx < len(lines) and _LIST_RE.match(lines[idx].strip()):
                    body.append(lines[idx])
                    idx += 1
                add_block("list", "\n".join(body), start + 1, idx, section_path)
                continue

            # 处理引用块（>）
            if stripped.startswith(">"):
                start = idx
                body = [stripped.lstrip(">").strip()]
                idx += 1
                while idx < len(lines) and lines[idx].strip().startswith(">"):
                    body.append(lines[idx].strip().lstrip(">").strip())
                    idx += 1
                add_block("blockquote", "\n".join(body), start + 1, idx, section_path)
                continue

            # 处理水平分割线
            if _HR_RE.match(stripped):
                add_block("hr", stripped, idx + 1, idx + 1, section_path)
                idx += 1
                continue

            # 处理普通段落（连续非空且非特殊行）
            start = idx
            body = [raw]
            idx += 1
            while idx < len(lines) and lines[idx].strip() and not self._is_special(lines[idx]):
                body.append(lines[idx])
                idx += 1
            add_block("paragraph", "\n".join(body), start + 1, idx, section_path)

        if not blocks:
            raise ValueError("no parseable content found")
        return parsed_documents(blocks, doc_id, str(parse_config.get("doc_name") or file_path.name))
