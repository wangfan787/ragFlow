"""文档入库流水线：解析 → 切分 → 元数据 → 索引。

这是把各阶段拼起来的组装层，放在 services 下，符合
"apps/services 负责把各阶段组装起来" 的分层意图。
"""

from __future__ import annotations

import logging

from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.markdown_chunker import MarkdownChunker
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.indexing.retrieval_metadata import RetrievalMetadataGenerator
from backend.src.parsing.models import ParseResultBlock, coerce_source_span, parse_source_from_config
from backend.src.parsing.parser_factory import build_parser

logger = logging.getLogger("mvp_api")


class IngestionPipeline:
    """MVP full-chain parsing/chunking pipeline (orchestration only)."""

    def __init__(
        self,
        *,
        embedding_indexer: EmbeddingIndexer | None = None,
        metadata_generator: RetrievalMetadataGenerator | None = None,
    ) -> None:
        self.chunker = MarkdownChunker()
        self.embedding_indexer = embedding_indexer or EmbeddingIndexer()
        self.metadata_generator = metadata_generator or RetrievalMetadataGenerator.from_env()

    def parse_stage(self, doc_id: str, parse_config: dict) -> list[ParseResultBlock]:
        source = parse_source_from_config(parse_config)
        parser = build_parser(source.source_type)
        records = parser.parse(doc_id, source)
        return [
            ParseResultBlock(
                text=item["text"],
                block_type=item["block_type"],
                page_no=item["page_no"],
                bbox=item.get("bbox"),
                section_path=item.get("section_path", []),
                order=int(item.get("order", index)),
                source_span=coerce_source_span(item.get("source_span")),
                metadata=dict(item.get("metadata", {})),
            )
            for index, item in enumerate(records, start=1)
        ]

    def chunk_stage(
        self,
        doc_id: str,
        parse_blocks: list[ParseResultBlock],
        chunk_config: ChunkConfig | dict,
        doc_name: str = "",
    ) -> list[dict]:
        config = build_chunk_config(chunk_config)
        chunk_inputs = []
        for block in parse_blocks:
            chunk_inputs.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "text": block.text,
                    "block_type": block.block_type,
                    "section_path": list(block.section_path),
                    "page_no": block.page_no,
                    "order": block.order,
                    "source_span": block.source_span,
                    "metadata": dict(block.metadata),
                }
            )

        rows = self.chunker.chunk(chunk_inputs, config)
        for row in rows:
            row["doc_id"] = doc_id
            row["doc_name"] = doc_name
            row.setdefault("content", row.get("text", ""))
            row["embedding_input_budget"] = config.embedding_input_budget
        return rows

    def metadata_stage(self, doc_name: str, chunks: list[dict]) -> list[dict]:
        rows: list[dict] = []
        source_counts: dict[str, int] = {}
        for chunk in chunks:
            metadata = (
                self.metadata_generator.generate(doc_name, chunk)
                if chunk.get("retrieval_eligible")
                else {
                    "important_kwd": [],
                    "important_tks": [],
                    "question_kwd": [],
                    "question_tks": [],
                    "title_tks": [],
                    "retrieval_metadata_trace": {"metadata_source": "context_parent_skipped"},
                }
            )
            source = str(
                metadata.get("retrieval_metadata_trace", {}).get(
                    "metadata_source",
                    "unknown",
                )
            )
            source_counts[source] = source_counts.get(source, 0) + 1
            rows.append({**chunk, **metadata})
        logger.info(
            "ingestion.metadata doc=%r chunks=%d sources=%s",
            doc_name,
            len(rows),
            source_counts,
        )
        return rows

    def run(
        self,
        doc_id: str,
        parse_config: dict,
        chunk_config: ChunkConfig | dict,
    ) -> dict:
        parsed = self.parse_stage(doc_id, parse_config)
        doc_name = str(parse_config.get("doc_name", ""))
        chunks = self.chunk_stage(doc_id, parsed, chunk_config, doc_name)
        chunks = self.metadata_stage(doc_name, chunks)

        indexed_count = self.embedding_indexer.index(
            doc_id,
            chunks,
            doc_name=doc_name,
        )
        return {
            "doc_id": doc_id,
            "parsed_count": len(parsed),
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
        }
