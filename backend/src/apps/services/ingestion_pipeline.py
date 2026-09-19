"""文档入库：解析 Document → 父子 Document → 正文嵌入与索引。

Markdown 源在解析后额外经过 ImagePipeline（P0-A）：本地相对路径图片
保存为资产并生成 VLM 描述的原子 image block；owner_id 从文档记录
透传，用于资产归属登记。
"""

from langchain_core.documents import Document

from backend.src.apps.services.image_pipeline import ImagePipeline
from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.block_chunker import BlockChunker
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.parsing.models import parse_source_from_config
from backend.src.parsing.parser_factory import build_parser

_MARKDOWN_TYPES = {"md", "markdown"}


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedding_indexer: EmbeddingIndexer | None = None,
        image_pipeline: ImagePipeline | None = None,
    ) -> None:
        self.chunker = BlockChunker()
        self.embedding_indexer = embedding_indexer or EmbeddingIndexer()
        self._image_pipeline = image_pipeline

    @property
    def image_pipeline(self) -> ImagePipeline:
        """惰性构造默认图片管线；VLM 模型在首次描述时才初始化。"""
        if self._image_pipeline is None:
            self._image_pipeline = ImagePipeline()
        return self._image_pipeline

    def parse_stage(
        self, doc_id: str, parse_config: dict, *, owner_id: str | None = None,
    ) -> list[Document]:
        file_type = parse_config.get("file_type") or parse_config.get("source_type", "")
        blocks = build_parser(file_type).parse(doc_id, parse_config)
        if file_type in _MARKDOWN_TYPES and parse_config.get("file_path"):
            source = parse_source_from_config(parse_config)
            blocks = self.image_pipeline.enrich_markdown(doc_id, owner_id, source, blocks)
        return blocks

    def chunk_stage(
        self, doc_id: str, parse_blocks: list[Document],
        chunk_config: ChunkConfig | dict, doc_name: str = "",
    ) -> list[Document]:
        config = build_chunk_config(chunk_config)
        chunks = self.chunker.chunk(parse_blocks, config)
        for chunk in chunks:
            chunk.metadata.update(doc_id=doc_id, doc_name=doc_name)
        return chunks

    def run(
        self,
        doc_id: str,
        parse_config: dict,
        chunk_config: ChunkConfig | dict,
        *,
        owner_id: str | None = None,
    ) -> dict:
        parsed = self.parse_stage(doc_id, parse_config, owner_id=owner_id)
        doc_name = str(parse_config.get("doc_name", ""))
        chunks = self.chunk_stage(doc_id, parsed, chunk_config, doc_name)
        indexed_count = self.embedding_indexer.index(doc_id, chunks, doc_name=doc_name)
        return {
            "doc_id": doc_id,
            "parsed_count": len(parsed),
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
            "image_trace": dict(self.image_pipeline.last_trace),
        }
