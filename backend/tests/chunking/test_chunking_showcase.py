"""用 T2Retrieval 子集里的真实文档，可视化 chunking 层的输入与输出。

运行（-s 才能看到打印）：
    pytest backend/tests/test_chunking_showcase.py -s -q

数据流：parsing 产出的 block 列表 → BlockChunker.chunk() → Parent(上下文) + Child(可召回)
重点展示四条生产不变量：token 硬上限、逐字守恒、父子链接、召回角色。
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter
from backend.src.chunking import BlockChunker, ChunkConfig
from backend.src.parsing.parser_factory import build_parser

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "dataset" / "T2Retrieval-subset" / "corpus.parquet"


def _pick_rows() -> list[tuple[dict, str]]:
    """与 parsing showcase 相同的选择逻辑：子集第一篇 HTML + 第一篇纯文本。"""
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


def _preview(text: str, limit: int = 90) -> str:
    return text.replace("\n", "⏎")[:limit] + ("…" if len(text) > limit else "")


def _show(row: dict, file_type: str, config: ChunkConfig) -> None:
    doc_id = str(row["_id"])
    text = str(row.get("text") or "").strip()
    blocks = build_parser(file_type).parse(
        doc_id, {"file_type": file_type, "text": text, "doc_name": doc_id}
    )

    print(f"\n{'=' * 76}")
    print(
        f"【输入】doc_id={doc_id}  {len(blocks)} 个 block"
        f"（{file_type} parser 产出，来自上一步 parsing）"
    )
    print(
        f"【配置】Parent target/max = {config.parent_target_tokens}/{config.parent_max_tokens}"
        f" | Child target/max = {config.child_target_tokens}/{config.child_max_tokens}"
    )

    docs = BlockChunker().chunk(blocks, config)
    parents = [d for d in docs if d.metadata["chunk_role"] == "parent"]
    children = [d for d in docs if d.metadata["chunk_role"] == "child"]
    by_parent: dict[str, list] = {}
    for child in children:
        by_parent.setdefault(child.metadata["parent_id"], []).append(child)

    child_tokens = [c.metadata["token_count"] for c in children]
    parent_tokens = [p.metadata["token_count"] for p in parents]
    print(
        f"【输出】{len(parents)} 个 Parent + {len(children)} 个 Child"
        f"（Parent 只存上下文不可召回，Child 唯一进入检索）"
    )
    if child_tokens:
        print(
            f"        Child token: max={max(child_tokens)} / mean={sum(child_tokens) // len(children)}"
            f"  |  Parent token: max={max(parent_tokens)}"
        )

    shown = 0
    for parent in parents[:2]:
        kids = sorted(by_parent[parent.metadata["chunk_id"]], key=lambda c: c.metadata["chunk_order"])
        conserved = "".join(k.page_content for k in kids) == parent.page_content
        print(f"  ── Parent {parent.metadata['chunk_order']}  token={parent.metadata['token_count']}"
              f"  child_ids={len(kids)} 个  守恒={'✓' if conserved else '✗'}")
        print(f"     正文: {_preview(parent.page_content, 80)}")
        for child in kids[:4]:
            cs, ce = child.metadata["parent_char_start"], child.metadata["parent_char_end"]
            sliced = parent.page_content[cs:ce] == child.page_content
            print(f"     ├─ Child {child.metadata['chunk_order']}  token={child.metadata['token_count']:>3}"
                  f"  parent字符 {cs}–{ce}  切片吻合={'✓' if sliced else '✗'}"
                  f"  『{_preview(child.page_content, 40)}』")
        if len(kids) > 4:
            print(f"     └─ …（其余 {len(kids) - 4} 个 Child 略）")
        shown += 1
    if len(parents) > shown:
        print(f"  …（其余 {len(parents) - shown} 个 Parent 略）")


def test_show_chunking_input_and_output_for_real_t2_docs() -> None:
    config = ChunkConfig()
    for row, file_type in _pick_rows():
        _show(row, file_type, config)

        doc_id = str(row["_id"])
        text = str(row.get("text") or "").strip()
        blocks = build_parser(file_type).parse(
            doc_id, {"file_type": file_type, "text": text, "doc_name": doc_id}
        )
        docs = BlockChunker().chunk(blocks, config)
        parents = [d for d in docs if d.metadata["chunk_role"] == "parent"]
        children = [d for d in docs if d.metadata["chunk_role"] == "child"]
        by_parent: dict[str, list] = {}
        for child in children:
            by_parent.setdefault(child.metadata["parent_id"], []).append(child)

        assert parents and children, "真实文档应同时产出 Parent 与 Child"
        for parent in parents:
            meta = parent.metadata
            assert meta["retrieval_eligible"] is False, "Parent 不可召回"
            assert meta["token_count"] <= config.parent_max_tokens, "Parent 不得超过硬上限"
            kids = sorted(by_parent[meta["chunk_id"]], key=lambda c: c.metadata["chunk_order"])
            assert [k.metadata["chunk_id"] for k in kids] == meta["child_ids"], "child_ids 必须完整"
            assert "".join(k.page_content for k in kids) == parent.page_content, \
                "Child 拼接必须逐字守恒 Parent 正文（生产不变量）"
            for child in kids:
                cm = child.metadata
                assert cm["retrieval_eligible"] is True, "Child 必须可召回"
                assert cm["token_count"] <= config.child_max_tokens, "Child 不得超过硬上限"
                assert parent.page_content[cm["parent_char_start"]:cm["parent_char_end"]] == child.page_content, \
                    "Child 必须是 Parent 文本的精确切片"
