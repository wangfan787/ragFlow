"""Markdown 图片 → 资产保存 → VLM 描述 → 原子 image block 的编排层。

只负责编排与失败处理（docs/plan/算法层QA.md §7.2 P0-A）：
- 原图按 SHA256 保存并登记归属，读取必须走 asset_id + owner 鉴权；
- VLM 失败时保留资产与 failed 状态，不伪造描述；
- skipped 引用（穿越/缺文件/类型不支持等）保留占位 block 与原因，
  保证位置与 order 连续，不静默消失。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from backend.src.assets.asset_store import AssetFileStore, AssetRegistry
from backend.src.assets.vlm_describer import (
    VLM_PROMPT_VERSION,
    DescriptionOutcome,
    ImageDescriber,
    assemble_page_content,
)
from backend.src.config.settings import settings
from backend.src.parsing.markdown_images import (
    ImageRef,
    extract_image_refs,
    interleave_image_blocks,
)
from backend.src.parsing.models import ParseSource

_MARKDOWN_TYPES = {"md", "markdown"}


def ref_alt(block: Document) -> str:
    return str(block.metadata.get("alt_text") or "")


class ImagePipeline:
    def __init__(
        self,
        describer: ImageDescriber | None = None,
        file_store: AssetFileStore | None = None,
        registry: AssetRegistry | None = None,
    ) -> None:
        self.describer = describer or ImageDescriber()
        self.file_store = file_store or AssetFileStore()
        self.registry = registry or AssetRegistry()
        # 仅供离线调试；正式链路的状态以 assets 表和 block metadata 为准
        self.last_trace: dict = {}

    def enrich_markdown(
        self,
        doc_id: str,
        owner_id: str | None,
        source: ParseSource,
        text_blocks: list[Document],
    ) -> list[Document]:
        """为 Markdown 解析结果补齐 image block；无图片或非文件来源时原样返回。"""
        if source.source_type not in _MARKDOWN_TYPES or source.file_path is None:
            return text_blocks
        refs = extract_image_refs(source.read_text(), source.file_path.parent)
        if not refs:
            return text_blocks
        image_blocks = [self._build_block(doc_id, owner_id, ref) for ref in refs]
        blocks = interleave_image_blocks(text_blocks, image_blocks)
        for block in image_blocks:
            section = list(block.metadata.get("section_path") or [])
            block.metadata["caption"] = section[-1] if section else (ref_alt(block) or "")
        statuses = [block.metadata.get("description_status", "skipped") for block in image_blocks]
        self.last_trace = {
            "image_ref_count": len(refs),
            "by_status": {status: statuses.count(status) for status in ("success", "failed", "skipped")},
        }
        return blocks

    def _build_block(self, doc_id: str, owner_id: str | None, ref: ImageRef) -> Document:
        base_metadata = {
            "block_type": "image",
            "page_no": None,
            "bbox": None,
            "section_path": [],  # interleave 时按位置回填
            "order": 0,
            "source_span": {
                "start_line": ref.start_line,
                "end_line": ref.end_line,
                "start_char": ref.start_char,
                "end_char": ref.end_char,
                "page_no": None,
                "accuracy": "exact",
            },
            "alt_text": ref.alt,
            "caption": "",  # 交错定位后按最近标题回填
        }
        if ref.skip_reason is not None:
            return self._skipped_block(ref, base_metadata)
        return self._described_block(doc_id, owner_id, ref, base_metadata)

    def _skipped_block(self, ref: ImageRef, base_metadata: dict) -> Document:
        placeholder = ref.alt or "无题注图片"
        reason = base_metadata.get("skip_reason") or ref.skip_reason or "unknown"
        metadata = {
            **base_metadata,
            "description_status": "skipped",
            "skip_reason": reason,
        }
        return Document(page_content=f"[图片] {placeholder}（未索引：{reason}）", metadata=metadata)

    def _described_block(
        self, doc_id: str, owner_id: str | None, ref: ImageRef, base_metadata: dict,
    ) -> Document:
        assert ref.resolved_path is not None and ref.mime_type is not None
        placeholder = ref.alt or "无题注图片"
        try:
            content = ref.resolved_path.read_bytes()
        except OSError:
            return self._skipped_block(ref, {**base_metadata, "skip_reason": "read_failed"})
        max_bytes = settings.integer("MVP_IMAGE_MAX_BYTES", 10 * 1024 * 1024, positive=True)
        if len(content) > max_bytes:
            return Document(
                page_content=f"[图片] {placeholder}（未索引：file_too_large）",
                metadata={**base_metadata, "description_status": "skipped", "skip_reason": "file_too_large"},
            )

        asset_id, storage_key = self.file_store.save(content, ref.resolved_path.suffix.lower())
        self.registry.register(
            asset_id=asset_id,
            document_id=doc_id,
            storage_key=storage_key,
            mime_type=ref.mime_type,
            sha256=asset_id.removeprefix("sha256:"),
            size_bytes=len(content),
            owner_id=owner_id,
        )

        outcome: DescriptionOutcome = self.describer.describe(
            content, ref.mime_type, alt_text=ref.alt,
        )
        self.registry.mark_description(
            asset_id=asset_id,
            document_id=doc_id,
            status=outcome.status,
            vlm_model=outcome.model_name,
            vlm_prompt_version=VLM_PROMPT_VERSION if outcome.status == "success" else None,
        )
        metadata = {
            **base_metadata,
            "asset_id": asset_id,
            "storage_key": storage_key,
            "mime_type": ref.mime_type,
            "size_bytes": len(content),
            "description_status": outcome.status,
            "vlm_model": outcome.model_name,
            "vlm_prompt_version": VLM_PROMPT_VERSION if outcome.status == "success" else None,
        }
        if outcome.status == "success" and outcome.description is not None:
            page_content = assemble_page_content(outcome.description, alt_text=ref.alt)
        else:
            # 资产已保存；描述失败如实占位，不编造内容
            page_content = f"[图片] {placeholder}（VLM 描述生成失败，原图资产已保留）"
        return Document(page_content=page_content, metadata=metadata)

