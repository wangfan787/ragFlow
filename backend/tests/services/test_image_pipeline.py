"""P0-A：ImagePipeline 编排契约（资产保存 / VLM 成败 / skip 原因 / 交错）。

验收对应 docs/plan/算法层QA.md §7.2 P0-A：
- 原图按 SHA256 保存并登记归属（owner 随文档传入）；
- VLM 失败保留资产与 failed 状态，不伪造描述；
- 重复图片共享同一 asset（SHA256 去重）；
- 穿越等非法引用保留 skipped 占位块。
"""

from __future__ import annotations

from pathlib import Path

from backend.src.apps.services.image_pipeline import ImagePipeline
from backend.src.assets.asset_store import AssetFileStore, AssetRegistry
from backend.src.assets.vlm_describer import DescriptionOutcome, ImageDescription
from backend.src.parsing.models import parse_source_from_config


class FakeDescriber:
    model_name = "fake-vlm-1"

    def __init__(self, behavior: str = "success") -> None:
        self.behavior = behavior
        self.calls: list[bytes] = []

    def describe(self, image_bytes: bytes, mime_type: str, *, alt_text: str = "", caption: str = "") -> DescriptionOutcome:
        self.calls.append(image_bytes)
        if self.behavior == "fail":
            return DescriptionOutcome(status="failed", model_name=self.model_name, error="vlm down")
        return DescriptionOutcome(
            status="success",
            model_name=self.model_name,
            description=ImageDescription(
                ocr_text="API → Retriever → Reranker",
                subjects=("API", "Retriever", "Reranker"),
                key_facts=("API 依次调用 Retriever 和 Reranker",),
                chart_trends="不适用",
                uncertainties=(),
            ),
        )


PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def _build_pipeline(tmp_path: Path, behavior: str = "success") -> tuple[ImagePipeline, AssetFileStore, AssetRegistry, FakeDescriber]:
    describer = FakeDescriber(behavior)
    store = AssetFileStore(root=tmp_path / "assets")
    registry = AssetRegistry(db_path=tmp_path / "db.sqlite3")
    return ImagePipeline(describer=describer, file_store=store, registry=registry), store, registry, describer


def _make_source(tmp_path: Path, markdown: str, images: dict[str, bytes] | None = None):
    for name, content in (images or {}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(content)
    md = tmp_path / "doc.md"
    md.write_text(markdown, encoding="utf-8")
    return parse_source_from_config({"file_type": "md", "file_path": str(md), "doc_name": "doc.md"})


def test_success_registers_asset_and_builds_described_block(tmp_path: Path) -> None:
    from langchain_core.documents import Document

    pipeline, store, registry, describer = _build_pipeline(tmp_path)
    source = _make_source(
        tmp_path,
        "# 检索流程\n\n混合检索先召回再重排。\n\n![架构图](images/arch.png)\n\n后续正文。\n",
        {"images/arch.png": PNG_BYTES},
    )
    heading = Document(
        page_content="检索流程",
        metadata={
            "block_type": "heading", "order": 1, "section_path": ["检索流程"],
            "source_span": {"start_line": 1, "end_line": 1, "start_char": 0, "end_char": 6,
                            "page_no": None, "accuracy": "line_only"},
        },
    )
    text_blocks = pipeline.enrich_markdown("doc_1", "alice", source, [heading])
    images_out = [block for block in text_blocks if block.metadata["block_type"] == "image"]
    assert len(images_out) == 1
    block = images_out[0]
    assert block.metadata["description_status"] == "success"
    assert block.metadata["asset_id"].startswith("sha256:")
    assert block.metadata["vlm_model"] == "fake-vlm-1"
    assert "API → Retriever → Reranker" in block.page_content
    assert block.metadata["section_path"] == ["检索流程"]
    assert block.metadata["caption"] == "检索流程"

    row = registry.get_asset(block.metadata["asset_id"], "doc_1")
    assert row is not None and row["owner_id"] == "alice"
    assert row["description_status"] == "success"
    assert store.read(row["storage_key"]) == PNG_BYTES
    assert describer.calls == [PNG_BYTES]


def test_duplicate_images_share_one_asset(tmp_path: Path) -> None:
    pipeline, store, registry, _ = _build_pipeline(tmp_path)
    source = _make_source(
        tmp_path,
        "# 图\n\n![一](a.png)\n\n中段文字\n\n![二](b.png)\n",
        {"a.png": PNG_BYTES, "b.png": PNG_BYTES},  # 内容完全相同 → 同一 SHA256
    )
    blocks = pipeline.enrich_markdown("doc_1", "alice", source, [])
    image_blocks = [block for block in blocks if block.metadata["block_type"] == "image"]
    asset_ids = {block.metadata["asset_id"] for block in image_blocks}
    assert len(asset_ids) == 1
    # 落盘只有一份；登记行也只有一个（同 asset 同 doc 幂等）
    stored = [path for path in (tmp_path / "assets").rglob("*") if path.is_file()]
    assert len(stored) == 1
    assert registry.get_asset(next(iter(asset_ids)), "doc_1")["size_bytes"] == len(PNG_BYTES)


def test_vlm_failure_keeps_asset_and_failed_status(tmp_path: Path) -> None:
    pipeline, store, registry, _ = _build_pipeline(tmp_path, behavior="fail")
    source = _make_source(tmp_path, "# 图\n\n![失败图](a.png)\n", {"a.png": PNG_BYTES})
    blocks = pipeline.enrich_markdown("doc_1", "alice", source, [])
    block = next(b for b in blocks if b.metadata["block_type"] == "image")
    assert block.metadata["description_status"] == "failed"
    assert "VLM 描述生成失败" in block.page_content
    # 资产保留：文件与登记行都在，状态如实为 failed
    row = registry.get_asset(block.metadata["asset_id"], "doc_1")
    assert row is not None and row["description_status"] == "failed"
    assert store.read(row["storage_key"]) == PNG_BYTES


def test_illegal_refs_leave_skipped_placeholders(tmp_path: Path) -> None:
    pipeline, _, registry, describer = _build_pipeline(tmp_path)
    source = _make_source(
        tmp_path,
        "# 图\n\n![穿越](../secret.png)\n\n![缺失](missing.png)\n\n![正常](ok.png)\n",
        {"ok.png": PNG_BYTES},
    )
    blocks = pipeline.enrich_markdown("doc_1", "alice", source, [])
    skipped = [
        block for block in blocks
        if block.metadata.get("description_status") == "skipped"
    ]
    assert [block.metadata["skip_reason"] for block in skipped] == ["path_traversal", "file_not_found"]
    assert all("asset_id" not in block.metadata for block in skipped)
    # 只有合法图片触发了 VLM
    assert len(describer.calls) == 1


def test_parse_stage_integrates_real_parser_and_pipeline(tmp_path: Path) -> None:
    """端到端：真实 UnstructuredParser 文本块 + image block 交错后 order 连续。"""
    from backend.src.apps.services.ingestion_pipeline import IngestionPipeline

    pipeline, _, _, _ = _build_pipeline(tmp_path)
    md = tmp_path / "doc.md"
    md.write_text(
        "# 检索流程\n\n混合检索先召回再重排。\n\n![架构图](arch.png)\n\n后续正文段落，用于验证交错位置。\n",
        encoding="utf-8",
    )
    (tmp_path / "arch.png").write_bytes(PNG_BYTES)
    ingestion = IngestionPipeline(image_pipeline=pipeline)
    blocks = ingestion.parse_stage(
        "doc_1",
        {"file_type": "md", "file_path": str(md), "doc_name": "doc.md"},
        owner_id="alice",
    )
    orders = [block.metadata["order"] for block in blocks]
    assert orders == list(range(1, len(blocks) + 1))
    image_index = next(i for i, block in enumerate(blocks) if block.metadata["block_type"] == "image")
    assert blocks[image_index].metadata["section_path"] == ["检索流程"]
    # image block 夹在正文段落之间，且自身 asset 已登记
    assert blocks[image_index].metadata["asset_id"].startswith("sha256:")


def test_ownerless_ingestion_still_registers_asset(tmp_path: Path) -> None:
    """离线/遗留文档没有 owner：资产照常保存，但归属为空（预览将拒绝）。"""
    pipeline, _, registry, _ = _build_pipeline(tmp_path)
    source = _make_source(tmp_path, "# 图\n\n![图](a.png)\n", {"a.png": PNG_BYTES})
    blocks = pipeline.enrich_markdown("doc_1", None, source, [])
    block = next(b for b in blocks if b.metadata["block_type"] == "image")
    row = registry.get_asset(block.metadata["asset_id"], "doc_1")
    assert row is not None and row["owner_id"] is None
    assert registry.authorized_asset(row["asset_id"], "doc_1", None) is None
