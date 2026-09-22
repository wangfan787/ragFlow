"""统一文档解析器：所有格式共用一个 unstructured 引擎实现。

旧方案为每种格式手写一个解析器（pypdf / BeautifulSoup / markdown-it-py），
格式覆盖与解析质量都受限于自研代码量。本模块改用 unstructured 作为统一
解析引擎：一个实现吃 md / pdf / html / txt，输出的 block 契约与旧解析器
完全一致（text / block_type / section_path / order / source_span ...），
下游 BlockChunker 无需感知引擎切换。

换引擎的已知取舍（面试可讲，对应 Tika 式"广格式 vs 格式内保真"权衡）：
- Markdown 代码/表格按语法边界原样保留，其余内容由 unstructured 解析；
- 其他格式的表格优先使用 text_as_html；引擎未提供结构时仅有纯文本；
- source_span 通过把 element 文本回定位到原文获得，Markdown 标记剥离后
  部分块退化为 line_only / unavailable 精度（与旧 HTML 解析器同策略）；
- PDF 走 fast 策略（pdfminer 纯文本抽取），不做 OCR 与版面模型；
- unstructured 的文本分类依赖 NLTK 数据（punkt_tab 与
  averaged_perceptron_tagger_eng），部署机需预下载。
"""
from __future__ import annotations

import html as _html
import io
import os
import re
import tempfile

from langchain_core.documents import Document

from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text

from .models import ParseSource, parse_source_from_config, parsed_documents
from .markdown_structures import prepare_markdown

# unstructured category → block 契约的 block_type；未列出的类别一律按
# paragraph 处理，保证 BlockMergeStrategy 的合并语义不变。
_CATEGORY_TO_BLOCK_TYPE = {
    "Title": "heading",
    "NarrativeText": "paragraph",
    "UncategorizedText": "paragraph",
    "Text": "paragraph",
    "Address": "paragraph",
    "Formula": "paragraph",
    "FigureCaption": "paragraph",
    "Header": "paragraph",
    "Footer": "paragraph",
    "ListItem": "list",
    "CodeSnippet": "code",
    "Code": "code",
    "Table": "table",
}

# 纯文本没有版面信号，unstructured 会把短句误判为 Title，全部按段落处理。
_PLAIN_TYPES = {"txt", "text"}


class _SpanLocator:
    """把 element 文本回定位到原文，给出保守的行号与精度。

    unstructured 对块文本做的规范化本质是「删除」：Markdown 标记字符与
    空白折叠。对原文做同样的删除（并保留每个保留字符的 raw 下标）后，
    element 文本必然是归一化原文的子串，单调游标查找即可无歧义回定位，
    不存在前缀猜测的误命中问题。精度语义与旧解析器一致：纯文本且 raw
    切片逐字一致才记 exact；md/html 文本经过规范化，一律 line_only。
    """

    # 空白 + Markdown 结构标记（含表格分隔行 --- 与 setext 下划线）；
    # 与 unstructured 的规范化方向保持一致，删除对两边同时生效。
    _NORMALIZE_RE = re.compile(r"[\s*`~_|>=#+-]+")

    def __init__(self, raw: str, allow_exact: bool = False) -> None:
        self._raw = raw
        self._allow_exact = allow_exact
        self._cursor = 0
        kept_chars: list[str] = []
        kept_indices: list[int] = []
        for index, char in enumerate(raw):
            if self._NORMALIZE_RE.match(char):
                continue
            kept_chars.append(char)
            kept_indices.append(index)
        self._normalized = "".join(kept_chars)
        self._raw_index = kept_indices

    def locate(self, text: str) -> dict:
        needle = self._NORMALIZE_RE.sub("", text)
        start = self._normalized.find(needle, self._cursor) if needle else -1
        if start < 0:
            return self._span(None, None, None, None, "unavailable")
        self._cursor = start + len(needle)
        raw_start = self._raw_index[start]
        raw_end = self._raw_index[start + len(needle) - 1] + 1
        start_line = self._raw.count("\n", 0, raw_start) + 1
        end_line = self._raw.count("\n", 0, raw_end) + 1
        exact = self._allow_exact and self._raw[raw_start:raw_end] == text
        accuracy = "exact" if exact else "line_only"
        return self._span(start_line, end_line, raw_start, raw_end, accuracy)

    @staticmethod
    def _span(start_line, end_line, start_char, end_char, accuracy) -> dict:
        return {
            "start_line": start_line,
            "end_line": end_line,
            "start_char": start_char,
            "end_char": end_char,
            "page_no": None,
            "accuracy": accuracy,
        }


class UnstructuredParser:
    """md / pdf / html / txt 共用的解析实现；格式差异只在 partition 调用。"""

    _TEXTUAL_TYPES = {"md", "markdown", "html", "htm", "txt", "text"}

    def parse(self, doc_id: str, parse_config: dict | ParseSource) -> list[Document]:
        source = (
            parse_config if isinstance(parse_config, ParseSource) else parse_source_from_config(parse_config)
        )
        file_type = source.source_type
        if file_type == "pdf":
            elements = self._partition_engine(
                lambda: self._partition_pdf(source)
            )
            blocks = self._to_blocks(elements, file_type, raw=None)
        elif file_type in self._TEXTUAL_TYPES:
            raw = source.read_text()  # 严格 UTF-8 校验统一在输入边界完成
            protected, frontmatter = {}, None
            if file_type in {"md", "markdown"}:
                partition_raw, protected, frontmatter = prepare_markdown(raw)
                elements = self._partition_engine(
                    lambda: partition_md(file=io.BytesIO(partition_raw.encode("utf-8")))
                )
            elif file_type in {"html", "htm"}:
                elements = self._partition_engine(lambda: partition_html(text=raw))
            else:
                elements = self._partition_engine(lambda: partition_text(text=raw))
            # md/html 的 element 文本已经过实体解码，回定位前原文需同样解码，
            # 否则 "Hello &amp; world" 与 "Hello & world" 永远匹配不上。
            locate_raw = (
                _html.unescape(raw) if file_type in {"md", "markdown", "html", "htm"} else raw
            )
            blocks = self._to_blocks(elements, file_type, locate_raw, protected)
            if frontmatter is not None:
                blocks.insert(0, frontmatter)
        else:
            raise ValueError(f"unsupported file_type for parser: {file_type}")

        if not blocks:
            raise ValueError("no parseable content found")
        return parsed_documents(blocks, doc_id, source.name)

    @staticmethod
    def _partition_engine(partition):  # noqa: ANN001 - unstructured element 列表
        """引擎对退化输入（如残缺标签 </pre>）会抛内部异常；按旧解析器契约
        把这类「引擎无法解析」归一为 ValueError("no parseable content found")，
        让上层按既有语义排除坏文档、不中断批量入库。部署类故障（NLTK 数据
        缺失的 LookupError、磁盘 OSError 等）保持原样抛出，绝不静默吞掉。"""
        try:
            return partition()
        except (LookupError, OSError, MemoryError):
            raise
        except Exception as exc:
            raise ValueError("no parseable content found") from exc

    @staticmethod
    def _partition_pdf(source: ParseSource) -> list:
        # unstructured.partition.pdf 会在 import 阶段初始化 numba 缓存。某些
        # 只读镜像/沙箱中的 site-packages 不能作为缓存目录，进而导致整个
        # FastAPI 在尚未解析 PDF 时就启动失败。PDF 依赖按需加载，并把缓存
        # 放到系统临时目录；调用方显式配置时仍尊重已有环境变量。
        os.environ.setdefault(
            "NUMBA_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "ragflow-numba-cache"),
        )
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "ragflow-matplotlib-cache"),
        )
        from unstructured.partition.pdf import partition_pdf

        file_path = source.file_path
        if file_path is None or not file_path.exists():
            raise FileNotFoundError(f"source file missing for pdf: {file_path}")
        return partition_pdf(filename=str(file_path), strategy="fast")

    def _to_blocks(
        self, elements: list, file_type: str, raw: str | None,
        protected: dict[str, dict] | None = None,
    ) -> list[dict]:
        blocks: list[dict] = []
        section_path: list[str] = []
        locator = (
            _SpanLocator(raw, allow_exact=file_type in _PLAIN_TYPES) if raw is not None else None
        )
        for element in elements:
            text = (element.text or "").strip()
            if not text:
                continue
            preserved = (protected or {}).get(text)
            block_type = (
                "paragraph" if file_type in _PLAIN_TYPES
                else _CATEGORY_TO_BLOCK_TYPE.get(element.category, "paragraph")
            )
            if preserved is not None:
                text = preserved["text"]
                block_type = preserved["block_type"]
            if block_type == "heading":
                depth = element.metadata.category_depth
                # category_depth 为 0 起算（h1→0）；缺失时按一级标题处理。
                level = depth + 1 if isinstance(depth, int) and depth >= 0 else 1
                section_path = section_path[: max(0, level - 1)]
                section_path.append(text)
            if locator is not None:
                span = locator.locate(_html.unescape(text) if preserved else text)
            else:
                span = {
                    "start_line": None,
                    "end_line": None,
                    "start_char": None,
                    "end_char": None,
                    "page_no": element.metadata.page_number,
                    "accuracy": "unavailable",
                }
            if preserved is not None:
                span = dict(preserved["source_span"])
            elif block_type == "table" and element.metadata.text_as_html:
                text = element.metadata.text_as_html
            elif block_type == "code":
                text = element.text
            blocks.append(
                {
                    "text": text,
                    "block_type": block_type,
                    "page_no": span.get("page_no"),
                    "bbox": None,
                    "section_path": list(section_path),
                    "order": len(blocks) + 1,
                    "source_span": span,
                    "metadata": {},
                }
            )
        return blocks
