"""父子切片：保留正文、来源范围和父块内坐标，输出 LangChain Document。"""

from dataclasses import asdict
import hashlib
import json

from langchain_core.documents import Document

from backend.src.chunking.block_merge import BlockMergeStrategy
from backend.src.chunking.chunk_config import ChunkConfig, build_chunk_config
from backend.src.chunking.token_counter import SimpleTokenCounter


class MarkdownChunker:
    def __init__(self) -> None:
        self._merge = BlockMergeStrategy()
        self._counter = SimpleTokenCounter()

    @staticmethod
    def _merge_span(spans: list[dict | None]) -> dict | None:
        spans = [span for span in spans if span]
        if not spans:
            return None
        result = {}
        for key in ("start_line", "end_line", "start_char", "end_char", "page_no"):
            ordered = reversed(spans) if key.startswith("end_") else spans
            result[key] = next((s[key] for s in ordered if s.get(key) is not None), None)
        accuracies = [span.get("accuracy", "unavailable") for span in spans]
        result["accuracy"] = (
            "exact" if all(value == "exact" for value in accuracies)
            else "line_only" if any(value in {"exact", "line_only"} for value in accuracies)
            else "unavailable"
        )
        return result

    def _child_span(self, item: dict) -> dict | None:
        start, end = item.get("parent_char_start"), item.get("parent_char_end")
        fallback = [block.metadata.get("source_span") for block in item.get("source_blocks", [])]
        if start is None or end is None:
            return self._merge_span(fallback)
        projected = []
        for segment in item.get("parent_segment_map", []):
            overlap_start = max(int(start), int(segment["parent_start"]))
            overlap_end = min(int(end), int(segment["parent_end"]))
            if overlap_start >= overlap_end:
                continue
            span = segment["block"].metadata.get("source_span")
            if not span:
                continue
            span = dict(span)
            if span.get("accuracy") == "exact" and span.get("start_char") is not None:
                span["start_char"] += int(segment["block_char_start"]) + overlap_start - int(segment["parent_start"])
                span["end_char"] = span["start_char"] + overlap_end - overlap_start
            else:
                span["accuracy"] = "line_only" if span.get("accuracy", "unavailable") != "unavailable" else "unavailable"
            projected.append(span)
        return self._merge_span(projected or fallback)

    def chunk(
        self, parse_blocks: list[Document], chunk_config: ChunkConfig | dict,
    ) -> list[Document]:
        config = build_chunk_config(chunk_config)
        profile_hash = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        merged = self._merge.merge(parse_blocks, config)
        doc_id = next(
            (str(block.metadata.get("doc_id", "doc_unknown")) for block in parse_blocks if block.page_content.strip()),
            "doc_unknown",
        )
        identified = [(order, item, f"{doc_id}_v2_{profile_hash}_{order}")
                      for order, item in enumerate(merged, start=1)]
        parents = {id(item): chunk_id for _, item, chunk_id in identified if item["chunk_role"] == "parent"}
        children: dict[str, list[str]] = {}
        for _, item, chunk_id in identified:
            if item["chunk_role"] == "child":
                parent_id = parents[id(item["parent_ref"])]
                item["parent_id"] = parent_id
                children.setdefault(parent_id, []).append(chunk_id)

        documents = []
        for order, item, chunk_id in identified:
            blocks = item.get("source_blocks", [])
            text = item.get("text", "")
            if not blocks or not text.strip():
                continue
            first = blocks[0].metadata
            chunk_doc_id = str(first.get("doc_id", doc_id))
            if not chunk_doc_id:
                raise ValueError("doc_id cannot be empty")
            role = item["chunk_role"]
            span = self._child_span(item) if role == "child" else self._merge_span(
                [block.metadata.get("source_span") for block in blocks]
            )
            documents.append(Document(
                page_content=text,
                metadata={
                    "chunk_id": chunk_id, "doc_id": chunk_doc_id,
                    "doc_name": first.get("doc_name", ""),
                    "section_path": list(first.get("section_path", [])),
                    "page_no": first.get("page_no"),
                    "chunk_role": role, "parent_id": item.get("parent_id"),
                    "child_ids": children.get(chunk_id, []), "chunk_order": order,
                    "chunk_profile_version": "parent-child-v2", "chunk_profile_hash": profile_hash,
                    "retrieval_eligible": role == "child",
                    "source_span": span,
                    "source_block_ids": list(dict.fromkeys(
                        f"{chunk_doc_id}_blk_{int(block.metadata.get('order', order))}" for block in blocks
                    )),
                    "parent_char_start": item.get("parent_char_start"),
                    "parent_char_end": item.get("parent_char_end"),
                    "block_types": [block.metadata.get("block_type", "paragraph") for block in blocks],
                    "token_count": self._counter.count(text),
                    "trace": {key: value for key, value in item.get("trace", {}).items() if key != "chunk_role"},
                    "embedding_input_budget": config.embedding_input_budget,
                },
            ))
        return documents
