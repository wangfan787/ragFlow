from __future__ import annotations

import logging

from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from backend.src.contracts import EmbeddingModel, SearchStore
from backend.src.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.rule_reranker import RuleReranker
from backend.src.retrieval.embedding_retriever import EmbeddingRetriever
from backend.src.retrieval.hybrid_fusion import HybridFusion
from backend.src.retrieval.keyword_retriever import KeywordRetriever
from backend.src.retrieval.models import RetrievedChunk, validate_retrieved_chunk
from backend.src.retrieval.vectorizer import query_terms

logger = logging.getLogger("mvp_api")


class HybridRouter:
    """Hybrid router orchestrating keyword + vector + optional rerank."""

    def __init__(
        self,
        store: SearchStore | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        shared_store = store or ElasticsearchStore()
        self.embedding_retriever = EmbeddingRetriever(
            embedding_model=embedding_model,
            store=shared_store,
        )
        self.keyword_retriever = KeywordRetriever(store=shared_store)
        self.fusion = HybridFusion()
        self.reranker = RuleReranker()
        self._cross_encoder: CrossEncoderReranker | None = None
        self.last_trace: dict = {}

    def _reranker_for_backend(self, backend: str):
        if backend != "cross-encoder":
            return self.reranker
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker()
        return self._cross_encoder

    def _expand_parent_context(
        self,
        eligible_rows: list[dict],
    ) -> tuple[list[dict], dict]:
        """Aggregate hit Children by Parent using RAGFlow's mean semantics."""
        if not eligible_rows:
            return eligible_rows, {"expanded": 0, "deduped": 0, "parent_lookup_failed": 0}

        # Parent records are context-only. Drop any legacy Parent candidate
        # defensively even if a custom store ignored the mandatory filter.
        children = [
            row
            for row in eligible_rows
            if str(row.get("chunk_role", "child")) == "child"
            and bool(row.get("retrieval_eligible", True))
        ]
        child_parent_ids = {str(row["parent_id"]) for row in children if row.get("parent_id")}
        parent_payloads: dict[str, dict] = {}
        if child_parent_ids:
            parent_rows = self.embedding_retriever.query_by_ids(list(child_parent_ids))
            for parent_record in parent_rows:
                payload = dict(parent_record.payload)
                pid = str(payload.get("chunk_id") or parent_record.id or "")
                if pid:
                    parent_payloads[pid] = payload
        failed_lookups = len(child_parent_ids) - len(parent_payloads)
        families: dict[str, list[dict]] = {}
        for row in children:
            family_id = str(row.get("parent_id") or row.get("chunk_id"))
            families.setdefault(family_id, []).append(row)

        expanded_rows: list[dict] = []
        expanded_count = 0
        for family_id, members in families.items():
            members = sorted(
                members,
                key=lambda row: float(row.get("score", row.get("fused_score", 0.0))),
                reverse=True,
            )
            primary = dict(members[0])

            def mean(field: str) -> float:
                return sum(float(row.get(field, 0.0) or 0.0) for row in members) / len(members)

            matched_children = [
                {
                    "chunk_id": str(row["chunk_id"]),
                    "score": float(row.get("score", row.get("fused_score", 0.0))),
                    "vector_score": float(row.get("vector_score", 0.0)),
                    "keyword_score": float(row.get("keyword_score", 0.0)),
                    "fused_score": float(row.get("fused_score", 0.0)),
                    "rerank_score": row.get("rerank_score"),
                    "snippet": str(row.get("content", "")),
                    "source_span": dict(row.get("source_span") or {}),
                    "source_block_ids": list(row.get("source_block_ids", []) or []),
                    "parent_char_start": row.get("parent_char_start"),
                    "parent_char_end": row.get("parent_char_end"),
                }
                for row in members
            ]
            family_score = mean("score")
            result = {
                **primary,
                "score": family_score,
                "vector_score": mean("vector_score"),
                "keyword_score": mean("keyword_score"),
                "fused_score": mean("fused_score"),
                "rerank_score": (
                    mean("rerank_score")
                    if any(row.get("rerank_score") is not None for row in members)
                    else None
                ),
                "matched_child_id": str(primary["chunk_id"]),
                "primary_matched_child_id": str(primary["chunk_id"]),
                "matched_children": matched_children,
                "family_contributors": matched_children,
                "family_member_count": len(members),
            }
            parent_payload = parent_payloads.get(family_id)
            if parent_payload and str(parent_payload.get("content", "")):
                result.update(
                    {
                        "chunk_id": str(parent_payload.get("chunk_id") or family_id),
                        "doc_id": str(parent_payload.get("doc_id") or primary["doc_id"]),
                        "content": str(parent_payload["content"]),
                        "doc_name": str(parent_payload.get("doc_name") or primary.get("doc_name", "")),
                        "section_path": list(parent_payload.get("section_path", [])),
                        "page_no": parent_payload.get("page_no"),
                        "chunk_role": "parent",
                        "retrieval_eligible": False,
                        "parent_id": None,
                        "child_ids": list(parent_payload.get("child_ids", []) or []),
                        "chunk_order": parent_payload.get("chunk_order"),
                        "source_span": dict(parent_payload.get("source_span") or {}),
                        "source_block_ids": list(parent_payload.get("source_block_ids", []) or []),
                        "context_source_block_ids": list(
                            parent_payload.get("source_block_ids", []) or []
                        ),
                        "context_span": dict(parent_payload.get("source_span") or {}),
                        "parent_char_start": None,
                        "parent_char_end": None,
                    }
                )
                expanded_count += 1
            else:
                # The family mean still uses every hit, but only the primary
                # Child text is available to the prompt when Parent lookup fails.
                result["matched_children"] = [matched_children[0]]
            expanded_rows.append(result)

        expanded_rows.sort(
            key=lambda r: (
                float(r.get("score", r.get("fused_score", 0.0))),
                float(r.get("keyword_score", 0.0)),
            ),
            reverse=True,
        )
        trace = {
            "expanded": expanded_count,
            "deduped": len(children) - len(expanded_rows),
            "parent_lookup_failed": failed_lookups,
            "legacy_parent_candidates_dropped": len(eligible_rows) - len(children),
            "family_score": "mean",
        }
        return expanded_rows, trace

    def retrieve(
        self,
        query: str,
        retrieval_config: RetrievalConfig | dict | None = None,
    ) -> list[RetrievedChunk]:
        config = build_retrieval_config(retrieval_config)

        vector_rows = self.embedding_retriever.retrieve(
            query,
            {
                "top_k": config.candidate_top_k,
                "filters": config.filters,
            },
        )
        keyword_rows = self.keyword_retriever.retrieve(
            query,
            {
                "top_k": config.candidate_top_k,
                "filters": config.filters,
            },
        )
        fused_rows = self.fusion.fuse(
            vector_rows,
            keyword_rows,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
        )

        ranked_rows = fused_rows
        fallback_reason: str | None = None
        if config.rerank_enabled:
            try:
                reranker = self._reranker_for_backend(config.rerank_backend)
                ranked_rows = reranker.rerank(query, fused_rows)
            except Exception as exc:  # pragma: no cover
                fallback_reason = f"rerank_failed:{exc}"
                logger.warning(
                    "retrieval.rerank_failed query=%r reason=%s",
                    query,
                    exc,
                    exc_info=True,
                )
                ranked_rows = fused_rows

        eligible_rows = [
            item
            for item in ranked_rows
            if float(item.get("score", item.get("fused_score", 0.0))) >= config.similarity_threshold
        ]

        # small-to-big + 父子去重：子块召回后展开为父块全文，同 family 只保留最高分。
        eligible_rows, expand_trace = self._expand_parent_context(eligible_rows)

        results: list[RetrievedChunk] = []
        for item in eligible_rows[: config.top_k]:
            score = float(item.get("score", item.get("fused_score", 0.0)))
            fused_score = float(item.get("fused_score", score))
            rerank_score = item.get("rerank_score")
            result = RetrievedChunk(
                chunk_id=str(item["chunk_id"]),
                doc_id=str(item["doc_id"]),
                content=str(item.get("content", "")),
                score=score,
                vector_score=float(item.get("vector_score", 0.0)),
                keyword_score=float(item.get("keyword_score", 0.0)),
                fused_score=fused_score,
                rerank_score=float(rerank_score) if rerank_score is not None else None,
                section_path=list(item.get("section_path", [])),
                page_no=item.get("page_no"),
                doc_name=str(item.get("doc_name", "")),
                important_kwd=list(item.get("important_kwd", [])),
                question_kwd=list(item.get("question_kwd", [])),
                embedding_backend=str(item.get("embedding_backend", "")),
                embedding_model=str(item.get("embedding_model", "")),
                embedding_dim=(
                    int(item["embedding_dim"]) if item.get("embedding_dim") is not None else None
                ),
                chunk_role=str(item.get("chunk_role", "parent")),
                parent_id=item.get("parent_id"),
                child_ids=list(item.get("child_ids", []) or []),
                chunk_order=item.get("chunk_order"),
                matched_child_id=item.get("matched_child_id"),
                primary_matched_child_id=item.get("primary_matched_child_id"),
                matched_children=list(item.get("matched_children", []) or []),
                family_contributors=list(item.get("family_contributors", []) or []),
                retrieval_eligible=bool(item.get("retrieval_eligible", False)),
                source_span=dict(item.get("source_span") or {}),
                source_block_ids=list(item.get("source_block_ids", []) or []),
                parent_char_start=item.get("parent_char_start"),
                parent_char_end=item.get("parent_char_end"),
                context_span=dict(item.get("context_span") or {}),
                context_source_block_ids=list(
                    item.get("context_source_block_ids", []) or []
                ),
            )
            validate_retrieved_chunk(result)
            results.append(result)

        vector_rank = {
            (str(item["doc_id"]), str(item["chunk_id"])): idx
            for idx, item in enumerate(vector_rows, start=1)
        }
        keyword_rank = {
            (str(item["doc_id"]), str(item["chunk_id"])): idx
            for idx, item in enumerate(keyword_rows, start=1)
        }
        fused_rank = {
            (str(item["doc_id"]), str(item["chunk_id"])): idx
            for idx, item in enumerate(fused_rows, start=1)
        }

        chunk_traces = []
        for idx, item in enumerate(eligible_rows[: config.top_k], start=1):
            retrieval_chunk_id = str(item.get("matched_child_id") or item["chunk_id"])
            key = (str(item["doc_id"]), retrieval_chunk_id)
            chunk_traces.append(
                {
                    "rank": idx,
                    "doc_id": str(item["doc_id"]),
                    "chunk_id": str(item["chunk_id"]),
                    "matched_child_id": item.get("matched_child_id"),
                    "matched_children": list(item.get("matched_children", []) or []),
                    "doc_name": str(item.get("doc_name", "")),
                    "rank_in_vector": vector_rank.get(key),
                    "rank_in_keyword": keyword_rank.get(key),
                    "rank_after_fusion": fused_rank.get(key),
                    "channel_hits": list(item.get("channel_hits", [])),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "vector_score": round(float(item.get("vector_score", 0.0)), 4),
                    "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                    "fused_score": round(float(item.get("fused_score", 0.0)), 4),
                    "rerank_score": (
                        round(float(item["rerank_score"]), 4)
                        if item.get("rerank_score") is not None
                        else None
                    ),
                    "section_path": list(item.get("section_path", [])),
                    "keyword_trace": item.get("keyword_trace"),
                    "important_kwd": list(item.get("important_kwd", [])),
                    "question_kwd": list(item.get("question_kwd", [])),
                    "retrieval_metadata_trace": item.get("retrieval_metadata_trace"),
                    "embedding_backend": item.get("embedding_backend"),
                    "embedding_model": item.get("embedding_model"),
                    "embedding_dim": item.get("embedding_dim"),
                    "snippet": str(item.get("content", "")).replace("\n", " ")[:120],
                }
            )

        self.last_trace = {
            "query_terms": sorted(query_terms(query)),
            "chunk_count": len(results),
            "avg_score": (
                round(sum(row.score for row in results) / len(results), 4) if results else 0.0
            ),
            "vector_candidate_count": len(vector_rows),
            "keyword_candidate_count": len(keyword_rows),
            "fused_candidate_count": len(fused_rows),
            "eligible_candidate_count": len(eligible_rows),
            "vector_weight": config.vector_weight,
            "keyword_weight": config.keyword_weight,
            "similarity_threshold": config.similarity_threshold,
            "rerank_enabled": config.rerank_enabled,
            "fallback_reason": fallback_reason,
            "parent_expansion": expand_trace,
            "chunk_traces": chunk_traces,
            "filtered_chunks": [
                {
                    "doc_id": str(item.get("doc_id", "")),
                    "chunk_id": str(item.get("chunk_id", "")),
                    "score": round(float(item.get("score", item.get("fused_score", 0.0))), 4),
                    "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                    "vector_score": round(float(item.get("vector_score", 0.0)), 4),
                }
                for item in ranked_rows
                if float(item.get("score", item.get("fused_score", 0.0)))
                < config.similarity_threshold
            ][:10],
        }
        logger.info(
            "retrieval.hybrid query=%r vector_candidates=%d "
            "keyword_candidates=%d fused_candidates=%d "
            "top=%s fallback_reason=%s",
            query,
            len(vector_rows),
            len(keyword_rows),
            len(fused_rows),
            chunk_traces[:5],
            fallback_reason,
        )
        return results
