"""P0-A：image block 的分块原子性与 asset_id 透传（chunk → citation）。

- image 是原子块：一图一块，不与相邻文本合并成同一父块；
- asset_id / description_status 等元数据从解析块透传到父块与子块；
- 命中 image 子块时 Citation 返回 asset_id，文本块不携带该键。
"""

from __future__ import annotations

from langchain_core.documents import Document

from backend.src.chunking import BlockChunker, ChunkConfig
from backend.src.citation.citation_service import CitationService


def _span(start: int, end: int, accuracy: str = "line_only") -> dict:
    return {
        "start_line": 1, "end_line": 1, "start_char": start, "end_char": end,
        "page_no": None, "accuracy": accuracy,
    }


def _block(block_type: str, text: str, order: int, span: tuple[int, int], **extra) -> Document:
    return Document(
        page_content=text,
        metadata={
            "block_type": block_type, "order": order, "section_path": ["架构"],
            "source_span": _span(*span), "doc_id": "doc_1",
            **extra,
        },
    )


def _parse_blocks() -> list[Document]:
    image_meta = {
        "asset_id": "sha256:" + "ab" * 32,
        "storage_key": "ab/" + "ab" * 32 + ".png",
        "mime_type": "image/png",
        "alt_text": "架构图",
        "caption": "架构",
        "description_status": "success",
        "vlm_model": "fake-vlm",
        "vlm_prompt_version": "image-desc-v1",
        "size_bytes": 128,
    }
    return [
        _block("heading", "检索流程", 1, (0, 8)),
        _block("paragraph", "混合检索先召回再重排。", 2, (10, 40)),
        _block(
            "image",
            "[图片] 架构图\n图内文字：API → Retriever\n关键事实：API 调用 Retriever",
            3, (45, 60), **image_meta,
        ),
        _block("paragraph", "后续正文段落，包含足够文字用于与图片块区分。", 4, (65, 110)),
    ]


def test_image_block_stays_atomic_in_chunking() -> None:
    chunks = BlockChunker().chunk(_parse_blocks(), ChunkConfig())
    image_parents = [
        chunk for chunk in chunks
        if chunk.metadata["chunk_role"] == "parent" and "image" in chunk.metadata["block_types"]
    ]
    assert len(image_parents) == 1
    # 一图一块：image 父块的 block_types 只能是 ["image"]，不得混入文本
    assert image_parents[0].metadata["block_types"] == ["image"]
    # 其余父块不含 image
    other_parents = [
        chunk for chunk in chunks
        if chunk.metadata["chunk_role"] == "parent" and "image" not in chunk.metadata["block_types"]
    ]
    assert other_parents
    # image 父块有自己的子块（描述进入 Embedding/检索）
    image_children = [
        chunk for chunk in chunks
        if chunk.metadata.get("parent_id") == image_parents[0].metadata["chunk_id"]
    ]
    assert image_children


def test_asset_metadata_passes_through_to_parent_and_child() -> None:
    chunks = BlockChunker().chunk(_parse_blocks(), ChunkConfig())
    carrying = [
        chunk for chunk in chunks
        if chunk.metadata.get("asset_id") == "sha256:" + "ab" * 32
    ]
    assert {chunk.metadata["chunk_role"] for chunk in carrying} == {"parent", "child"}
    for chunk in carrying:
        assert chunk.metadata["description_status"] == "success"
        assert chunk.metadata["mime_type"] == "image/png"
        assert chunk.metadata["vlm_prompt_version"] == "image-desc-v1"
    # 普通文本块不新增空键
    plain = next(
        chunk for chunk in chunks
        if chunk.metadata["chunk_role"] == "parent" and "image" not in chunk.metadata["block_types"]
    )
    assert "asset_id" not in plain.metadata
    assert "description_status" not in plain.metadata


def test_citation_returns_asset_id_for_image_hit() -> None:
    chunks = BlockChunker().chunk(_parse_blocks(), ChunkConfig())
    image_child = next(
        chunk for chunk in chunks
        if chunk.metadata.get("asset_id") and chunk.metadata["chunk_role"] == "child"
    )
    text_child = next(
        chunk for chunk in chunks
        if not chunk.metadata.get("asset_id") and chunk.metadata["chunk_role"] == "child"
    )
    image_child.metadata["score"] = 0.8
    text_child.metadata["score"] = 0.6
    payload = CitationService().build("架构图见 [1]，流程见 [2]。", [image_child, text_child])
    image_citation = next(c for c in payload["citations"] if c["citation_index"] == 1)
    text_citation = next(c for c in payload["citations"] if c["citation_index"] == 2)
    assert image_citation["asset_id"] == "sha256:" + "ab" * 32
    assert image_citation["description_status"] == "success"
    assert "asset_id" not in text_citation
