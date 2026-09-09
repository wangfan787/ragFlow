"""Dataset adapter: reuse production parsing, chunking and embedding input validation."""

import re

from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.chunking.block_chunker import BlockChunker
from backend.src.indexing.embedding_text_builder import EmbeddingTextBuilder
from backend.src.parsing.parser_factory import build_parser


class ProductionRagAdapter:
    _HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")

    def __init__(
        self, chunk_config: ChunkConfig | None = None,
        text_builder: EmbeddingTextBuilder | None = None,
    ) -> None:
        self.chunk_config = chunk_config or ChunkConfig()
        self.chunker = BlockChunker()
        self.text_builder = text_builder or EmbeddingTextBuilder()

    def child_inputs(self, *, doc_id: str, doc_name: str, text: str) -> list[dict]:
        file_type = "html" if self._HTML_RE.search(text) else "text"
        blocks = build_parser(file_type).parse(
            doc_id, {"file_type": file_type, "text": text, "doc_name": doc_name},
        )
        rows = []
        for chunk in self.chunker.chunk(blocks, self.chunk_config):
            metadata = chunk.metadata
            if not metadata["retrieval_eligible"]:
                continue
            built = self.text_builder.build(chunk)
            fields = (
                "chunk_id doc_id parent_id chunk_order source_span parent_char_start "
                "parent_char_end chunk_profile_version chunk_profile_hash"
            ).split()
            rows.append({
                **{field: metadata[field] for field in fields},
                "embedding_profile": built.profile,
                "embedding_input_tokens": built.token_count,
                "embedding_text": built.text,
            })
        return rows
