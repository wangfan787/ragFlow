"""文档入库：解析 Document → 父子 Document → 正文嵌入与索引。"""

from langchain_core.documents import Document

from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.block_chunker import BlockChunker
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.parsing.parser_factory import build_parser


class IngestionPipeline:
    def __init__(self, *, embedding_indexer: EmbeddingIndexer | None = None) -> None:
        self.chunker = BlockChunker()
        self.embedding_indexer = embedding_indexer or EmbeddingIndexer()

    def parse_stage(self, doc_id: str, parse_config: dict) -> list[Document]:
        file_type = parse_config.get("file_type") or parse_config.get("source_type", "")
        return build_parser(file_type).parse(doc_id, parse_config)

    def chunk_stage(
        self, doc_id: str, parse_blocks: list[Document],
        chunk_config: ChunkConfig | dict, doc_name: str = "",
    ) -> list[Document]:
        config = build_chunk_config(chunk_config)
        chunks = self.chunker.chunk(parse_blocks, config)
        for chunk in chunks:
            chunk.metadata.update(doc_id=doc_id, doc_name=doc_name)
        return chunks

    def run(self, doc_id: str, parse_config: dict, chunk_config: ChunkConfig | dict) -> dict:
        parsed = self.parse_stage(doc_id, parse_config)
        doc_name = str(parse_config.get("doc_name", ""))
        chunks = self.chunk_stage(doc_id, parsed, chunk_config, doc_name)
        indexed_count = self.embedding_indexer.index(doc_id, chunks, doc_name=doc_name)
        return {
            "doc_id": doc_id,
            "parsed_count": len(parsed),
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
        }
