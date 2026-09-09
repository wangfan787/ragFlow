"""用 T2Retrieval 子集里的真实文档，可视化 parsing 层的输入与输出。

运行（-s 才能看到打印）：
    pytest backend/tests/test_parsing_showcase.py -s -q

数据流：原始文本 → ParseSource（统一输入）→ build_parser 按类型路由
       → parser.parse() → list[Document]（统一 block 输出）→ 下游 BlockChunker
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter
from backend.src.parsing.models import ParseSource
from backend.src.parsing.parser_factory import build_parser

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "dataset" / "T2Retrieval-subset" / "corpus.parquet"
ACCURACY_LEVELS = {"exact", "line_only", "unavailable"}


def _pick_rows() -> list[tuple[dict, str]]:
    """从子集里各选一篇真实 HTML 文档与纯文本文档（路由规则与生产一致）。"""
    adapter = ProductionRagAdapter()
    picks: list[tuple[dict, str]] = []
    for row in pq.read_table(CORPUS).to_pylist():
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        file_type = "html" if adapter._HTML_RE.search(text) else "text"
        if file_type == "html" and not any(kind == "html" for _, kind in picks):
            picks.append((row, file_type))
        if file_type == "text" and not any(kind == "text" for _, kind in picks):
            picks.append((row, file_type))
        if len(picks) == 2:
            break
    return picks


def _preview(text: str, limit: int = 150) -> str:
    text = text.replace("\n", "⏎")
    return text[:limit] + ("…" if len(text) > limit else "")


def _show(row: dict, file_type: str) -> None:
    doc_id = str(row["_id"])
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()

    print(f"\n{'=' * 76}")
    print(f"【输入】doc_id={doc_id}  file_type={file_type}  title={title!r}  共 {len(text)} 字符")
    print(f"原文开头: {_preview(text)}")

    source = ParseSource(source_type=file_type, name=doc_id, text=text)
    blocks = build_parser(file_type).parse(doc_id, source)

    kinds: dict[str, int] = {}
    for doc in blocks:
        kinds[doc.metadata["block_type"]] = kinds.get(doc.metadata["block_type"], 0) + 1
    print(f"【输出】{len(blocks)} 个 block（LangChain Document）| block_type 分布: {kinds}")
    print(f"  首个 block 的全部 metadata 键: {sorted(blocks[0].metadata)}")
    for index, doc in enumerate(blocks[:5], start=1):
        meta = doc.metadata
        span = meta.get("source_span") or {}
        print(f"  ── block #{index}  type={meta.get('block_type')!r}  order={meta.get('order')}  "
              f"section_path={meta.get('section_path')}")
        print(f"     span: 行 {span.get('start_line')}–{span.get('end_line')}  |  "
              f"字符 {span.get('start_char')}–{span.get('end_char')}  |  accuracy={span.get('accuracy')}")
        print(f"     正文: {_preview(doc.page_content, 100)}")
    if len(blocks) > 5:
        print(f"  …（其余 {len(blocks) - 5} 个 block 略）")


def test_show_parsing_input_and_output_for_real_t2_docs() -> None:
    picks = _pick_rows()
    assert len(picks) == 2, "子集里应同时存在 HTML 与纯文本文档"

    for row, file_type in picks:
        _show(row, file_type)

        # 结构断言：输出是下游 BlockChunker 可直接消费的统一 block 契约
        doc_id = str(row["_id"])
        text = str(row.get("text") or "").strip()
        blocks = build_parser(file_type).parse(
            doc_id, ParseSource(source_type=file_type, name=doc_id, text=text)
        )
        assert blocks, "至少应产出一个 block"
        for doc in blocks:
            assert doc.page_content.strip(), "block 正文不允许为空"
            assert doc.metadata["block_type"], "block_type 是下游分块的核心字段"
            span = doc.metadata.get("source_span") or {}
            assert span.get("accuracy") in ACCURACY_LEVELS
            if span.get("start_char") is not None and file_type == "text":
                assert 0 <= span["start_char"] <= span["end_char"] <= len(text), \
                    "纯文本的字符 span 必须落在原文范围内"
