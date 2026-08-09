# 基于 markdown-it-py 的 Markdown 解析器（改进版）
from __future__ import annotations

from pathlib import Path
from markdown_it import MarkdownIt
from markdown_it.token import Token


class MarkdownParserIt:
    """基于 markdown-it-py 的 Markdown 解析器。

    优点：
    - 完全兼容 CommonMark 和 GFM 标准
    - 支持更多 Markdown 特性（嵌套列表、表格、删除线等）
    - 更健壮，边缘情况处理更好
    - 活跃维护的标准实现
    """

    def __init__(self) -> None:
        """初始化 markdown-it 解析器，启用 GFM 表格支持。"""
        self._md = MarkdownIt().enable("table")

    def parse(self, doc_id: str, parse_config: dict) -> list[dict]:
        """解析 Markdown 文件，输出结构化 block 列表。

        参数：
            doc_id: 文档 ID
            parse_config: 解析配置，包含 file_path 等信息

        返回：
            结构化 block 列表，每个 block 包含：
            - text: 文本内容
            - block_type: 类型（heading/code/table/list等）
            - section_path: 章节路径
            - order: 顺序号
            - source_span: 源文件位置
        """
        file_path = Path(parse_config.get("file_path", ""))
        if not file_path.exists():
            raise FileNotFoundError(f"source file missing for doc {doc_id}: {file_path}")

        # 读取文件内容
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        # 先手动提取 frontmatter（markdown-it 不原生支持）
        blocks: list[dict] = []
        order = 0
        section_path: list[str] = []
        start_idx = 0

        # 检测 frontmatter
        if len(lines) >= 3 and lines[0].strip() == "---":
            end = 1
            while end < len(lines) and lines[end].strip() != "---":
                end += 1
            if end < len(lines) and end > 1:
                order += 1
                blocks.append({
                    "text": "\n".join(lines[1:end]),
                    "block_type": "frontmatter",
                    "page_no": None,
                    "bbox": None,
                    "section_path": [],
                    "order": order,
                    "source_span": {
                        "start_line": 1,
                        "end_line": end + 1,
                        "start_char": 0,
                        "end_char": len("\n".join(lines[1:end])),
                        "page_no": None,
                    },
                    "metadata": {},
                })
                start_idx = end + 1

        # 如果整个文件都是 frontmatter，直接返回
        if start_idx >= len(lines):
            return blocks

        # 处理剩余内容（跳过 frontmatter 后的部分）
        remaining_text = "\n".join(lines[start_idx:])
        if not remaining_text.strip():
            return blocks

        # 使用 markdown-it 解析剩余内容
        tokens = self._md.parse(remaining_text)

        # 转换为我们的 block 格式
        i = 0
        base_line = start_idx + 1  # 调整行号基数

        while i < len(tokens):
            token = tokens[i]
            token_type = token.type

            # 跳过空白和文档边界标记
            if token_type in {"_blank", "soft_break", "hard_break"}:
                i += 1
                continue

            # 处理标题
            if token_type == "heading_open":
                level = int(token.tag[1])  # h1 → 1, h2 → 2, ...
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    title = tokens[i].content
                    # 更新章节路径
                    section_path = section_path[: max(0, level - 1)]
                    section_path.append(title)

                    order += 1
                    blocks.append({
                        "text": title,
                        "block_type": "heading",
                        "page_no": None,
                        "bbox": None,
                        "section_path": list(section_path),
                        "order": order,
                        "source_span": {
                            "start_line": (token.map[0] if token.map else 0) + base_line,
                            "end_line": (token.map[1] if token.map else 0) + base_line,
                            "start_char": 0,
                            "end_char": len(title),
                            "page_no": None,
                        },
                        "metadata": {},
                    })

            # 处理代码块（```fence）
            elif token_type == "fence":
                content = token.content
                order += 1
                blocks.append({
                    "text": content,
                    "block_type": "code",
                    "page_no": None,
                    "bbox": None,
                    "section_path": list(section_path),
                    "order": order,
                    "source_span": {
                        "start_line": (token.map[0] if token.map else 0) + base_line,
                        "end_line": (token.map[1] if token.map else 0) + base_line,
                        "start_char": 0,
                        "end_char": len(content),
                        "page_no": None,
                    },
                    "metadata": {},
                })

            # 处理表格（直接从原始文本提取）
            elif token_type == "table_open":
                # 从原始文本中提取表格
                start_line_num = (token.map[0] if token.map else 0) + base_line
                end_line_num = start_line_num

                # 找到表格结束位置
                j = i + 1
                while j < len(tokens) and tokens[j].type != "table_close":
                    if tokens[j].map and tokens[j].map[1] > 0:
                        end_line_num = max(end_line_num, (tokens[j].map[1] if tokens[j].map else 0) + base_line)
                    j += 1
                if j < len(tokens) and tokens[j].map:
                    end_line_num = max(end_line_num, (tokens[j].map[1] if tokens[j].map else 0) + base_line)

                # 从原始行中提取表格
                table_lines = []
                for line_idx in range(start_line_num - base_line, min(end_line_num - base_line + 1, len(lines))):
                    if line_idx < len(lines) and lines[line_idx].strip().startswith("|"):
                        table_lines.append(lines[line_idx])

                table_text = "\n".join(table_lines)
                if table_text.strip():
                    order += 1
                    blocks.append({
                        "text": table_text,
                        "block_type": "table",
                        "page_no": None,
                        "bbox": None,
                        "section_path": list(section_path),
                        "order": order,
                        "source_span": {
                            "start_line": start_line_num,
                            "end_line": end_line_num,
                            "start_char": 0,
                            "end_char": len(table_text),
                            "page_no": None,
                        },
                        "metadata": {},
                    })

                i = j  # 跳到 table_close

            # 处理列表
            elif token_type in {"bullet_list_open", "ordered_list_open"}:
                # 收集列表项
                list_items = []
                list_type = "bullet" if token_type == "bullet_list_open" else "ordered"
                list_start_line = (token.map[0] if token.map else 0) + base_line
                list_end_line = list_start_line

                j = i + 1
                while j < len(tokens):
                    sub_token = tokens[j]
                    if sub_token.type in {"bullet_list_close", "ordered_list_close"}:
                        break

                    # 收集列表项内容
                    if sub_token.type == "list_item_open":
                        item_lines = []
                        k = j + 1
                        while k < len(tokens) and tokens[k].type != "list_item_close":
                            if tokens[k].type == "inline":
                                content = tokens[k].content.strip()
                                if content:
                                    # 添加列表标记
                                    marker = "-" if list_type == "bullet" else "1."
                                    item_lines.append(f"{marker} {content}")
                            k += 1

                        if item_lines:
                            list_items.extend(item_lines)
                            if tokens[k - 1].map:
                                list_end_line = max(list_end_line, (tokens[k - 1].map[1] if tokens[k - 1].map else 0) + base_line)
                        j = k + 1
                    else:
                        j += 1

                list_text = "\n".join(list_items)
                if list_text.strip():
                    order += 1
                    blocks.append({
                        "text": list_text,
                        "block_type": "list",
                        "page_no": None,
                        "bbox": None,
                        "section_path": list(section_path),
                        "order": order,
                        "source_span": {
                            "start_line": list_start_line,
                            "end_line": list_end_line,
                            "start_char": 0,
                            "end_char": len(list_text),
                            "page_no": None,
                        },
                        "metadata": {},
                    })

                i = j  # 跳到 list_close

            # 处理引用块
            elif token_type == "blockquote_open":
                # 收集引用内容
                quote_lines = []
                quote_start_line = (token.map[0] if token.map else 0) + base_line
                quote_end_line = quote_start_line

                j = i + 1
                while j < len(tokens) and tokens[j].type != "blockquote_close":
                    if tokens[j].type == "inline":
                        quote_lines.append(tokens[j].content)
                        if tokens[j].map:
                            quote_end_line = max(quote_end_line, (tokens[j].map[1] if tokens[j].map else 0) + base_line)
                    j += 1

                quote_text = "\n".join(quote_lines)
                if quote_text.strip():
                    order += 1
                    blocks.append({
                        "text": quote_text,
                        "block_type": "blockquote",
                        "page_no": None,
                        "bbox": None,
                        "section_path": list(section_path),
                        "order": order,
                        "source_span": {
                            "start_line": quote_start_line,
                            "end_line": quote_end_line,
                            "start_char": 0,
                            "end_char": len(quote_text),
                            "page_no": None,
                        },
                        "metadata": {},
                    })

                i = j  # 跳到 blockquote_close

            # 处理水平分割线
            elif token_type == "hr":
                order += 1
                blocks.append({
                    "text": "---",
                    "block_type": "hr",
                    "page_no": None,
                    "bbox": None,
                    "section_path": list(section_path),
                    "order": order,
                    "source_span": {
                        "start_line": (token.map[0] if token.map else 0) + base_line,
                        "end_line": (token.map[1] if token.map else 0) + base_line,
                        "start_char": 0,
                        "end_char": 3,
                        "page_no": None,
                    },
                    "metadata": {},
                })

            # 处理段落
            elif token_type == "paragraph_open":
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    content = tokens[i].content
                    if content.strip():
                        order += 1
                        blocks.append({
                            "text": content,
                            "block_type": "paragraph",
                            "page_no": None,
                            "bbox": None,
                            "section_path": list(section_path),
                            "order": order,
                            "source_span": {
                                "start_line": ((token.map[0] if token.map else 0) + base_line),
                                "end_line": ((token.map[1] if token.map else 0) + base_line),
                                "start_char": 0,
                                "end_char": len(content),
                                "page_no": None,
                            },
                            "metadata": {},
                        })

            i += 1

        if not blocks:
            raise ValueError("no parseable content found")

        return blocks
