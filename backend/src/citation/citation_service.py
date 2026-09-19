"""从实际发送的 Document 生成引用，直接返回 JSON 字典。"""
import re

from langchain_core.documents import Document


class CitationService:
    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def build(self, answer: str, chunks: list[Document]) -> dict:
        indexes = list(dict.fromkeys(int(match.group(1)) for match in self._CITATION_PATTERN.finditer(answer)))
        referenced = [index for index in indexes if 1 <= index <= len(chunks)]
        invalid = [index for index in indexes if index not in referenced]
        citations = []
        for index in referenced:
            chunk = chunks[index - 1]
            metadata = chunk.metadata
            snippet = " ".join(chunk.page_content.split())
            if len(snippet) > 160:
                snippet = f"{snippet[:160].rstrip()}..."
            if not metadata.get("doc_id") or not metadata.get("chunk_id") or not snippet:
                raise ValueError("citation must include doc_id, chunk_id and evidence text")
            citations.append({
                "citation_index": index,
                "chunk_id": metadata["chunk_id"], "doc_id": metadata["doc_id"],
                "snippet": snippet, "page_no": metadata.get("page_no"),
                "section_path": list(metadata.get("section_path", [])),
                "context_chunk_id": metadata["chunk_id"], "context_snippet": snippet,
                "context_span": dict(metadata.get("context_span") or {}),
                "prompt_span": dict(metadata.get("prompt_span") or {}),
                "matched_children": list(metadata.get("matched_children") or []),
                "primary_matched_child_id": metadata.get("primary_matched_child_id") or metadata.get("matched_child_id"),
                "source_span": dict(metadata.get("source_span") or {}),
            })
            # image 命中块透传资产引用：前端凭 asset_id 走鉴权预览取原图
            if metadata.get("asset_id") is not None:
                citations[-1]["asset_id"] = str(metadata["asset_id"])
                citations[-1]["description_status"] = metadata.get("description_status")
        average = round(
            sum(chunks[index - 1].metadata["score"] for index in referenced) / len(referenced), 4,
        ) if referenced else 0.0
        return {
            "answer": answer, "citations": citations,
            "trace": {
                "citation_count": len(citations), "evidence_count": len(chunks),
                "referenced_indexes": referenced, "invalid_indexes": invalid,
                "citation_avg_score": average,
            },
        }
