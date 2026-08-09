# Chunking module: parent-child dual-granularity document chunking.
from backend.src.chunking.markdown_chunker import MarkdownChunker
from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.models import ChunkMeta, ChunkRecord, validate_chunk_record
from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.chunking.block_merge import BlockMergeStrategy

__all__ = [
    "MarkdownChunker",
    "ChunkConfig",
    "build_chunk_config",
    "ChunkMeta",
    "ChunkRecord",
    "validate_chunk_record",
    "SimpleTokenCounter",
    "BlockMergeStrategy",
]
