"""Dataset adapters that reuse the production parser/chunker/text builder."""

from __future__ import annotations

import re

from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.chunking.markdown_chunker import MarkdownChunker
from backend.src.indexing.embedding_text_builder import EmbeddingTextBuilder
from backend.src.indexing.retrieval_metadata import RetrievalMetadataGenerator
from backend.src.parsing.parser_factory import build_parser


class ProductionRagAdapter:
    """Map an in-memory document into production Child embedding inputs."""

    _HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")

    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        text_builder: EmbeddingTextBuilder | None = None,
        metadata_generator: RetrievalMetadataGenerator | None = None,
    ) -> None:
        self.chunk_config = chunk_config or ChunkConfig()
        self.chunker = MarkdownChunker()
        self.text_builder = text_builder or EmbeddingTextBuilder()
        # Benchmarks use the production deterministic fallback by default;
        # they never trigger an unbounded metadata-LLM bill implicitly.
        self.metadata_generator = metadata_generator or RetrievalMetadataGenerator()

    def child_inputs(self, *, doc_id: str, doc_name: str, text: str) -> list[dict]:
        file_type = "html" if self._HTML_RE.search(text) else "text"
        parser = build_parser(file_type)
        parsed = parser.parse(
            doc_id,
            {"file_type": file_type, "text": text, "doc_name": doc_name},
        )
        chunk_inputs = [
            {
                **block,
                "doc_id": doc_id,
                "doc_name": doc_name,
            }
            for block in parsed
        ]
        chunks = self.chunker.chunk(chunk_inputs, self.chunk_config)
        rows: list[dict] = []
        for chunk in chunks:
            if not chunk["retrieval_eligible"]:
                continue
            chunk["doc_name"] = doc_name
            chunk["embedding_input_budget"] = self.chunk_config.embedding_input_budget
            chunk.update(self.metadata_generator.generate(doc_name, chunk))
            built = self.text_builder.build(chunk, doc_name=doc_name)
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "parent_id": chunk["parent_id"],
                    "chunk_order": chunk["chunk_order"],
                    "source_span": chunk["source_span"],
                    "parent_char_start": chunk["parent_char_start"],
                    "parent_char_end": chunk["parent_char_end"],
                    "chunk_profile_version": chunk["chunk_profile_version"],
                    "chunk_profile_hash": chunk["chunk_profile_hash"],
                    "embedding_profile": f"{built.profile}+rule-metadata-v1",
                    "embedding_input_tokens": built.token_count,
                    "embedding_text": built.text,
                }
            )
        return rows
