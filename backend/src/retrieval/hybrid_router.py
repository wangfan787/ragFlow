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
        """small-to-big + 父子去重。

        1. 对每个命中的 child chunk，用 parent_id 反查父块全文，把 content 展开成父块内容
           （子块负责精准召回，父块提供完整上下文给 QA）。
        2. 同一 family（父+子）若都命中，只保留分数高的那个，避免 QA 看到重复证据。

        返回 (展开去重后的 rows, trace 信息)。
        """
        if not eligible_rows:
            return eligible_rows, {"expanded": 0, "deduped": 0, "parent_lookup_failed": 0}

        # 收集需要反查的 parent_id（仅 child 才有 parent_id）
        child_parent_ids = {
            str(row.get("parent_id"))
            for row in eligible_rows
            if str(row.get("chunk_role", "parent")) == "child" and row.get("parent_id")
        }
        parent_payloads: dict[str, dict] = {}
        failed_lookups = 0
        if child_parent_ids:
            parent_rows = self.embedding_retriever.query_by_ids(list(child_parent_ids))
            for parent_record in parent_rows:
                payload = dict(parent_record.payload)
                pid = str(payload.get("chunk_id") or parent_record.id or "")
                if pid:
                    parent_payloads[pid] = payload
            failed_lookups = len(child_parent_ids) - len(parent_payloads)

        # family key：child 用 parent_id 归族；parent（无 parent_id）用自身 chunk_id 归族
        def _family_key(row: dict) -> str:
            role = str(row.get("chunk_role", "parent"))
            if role == "child" and row.get("parent_id"):
                return f"family::{row['parent_id']}"
            return f"family::{row.get('chunk_id')}"

        families: dict[str, list[dict]] = {}
        for row in eligible_rows:
            families.setdefault(_family_key(row), []).append(row)

        expanded_rows: list[dict] = []
        expanded_count = 0
        deduped_count = 0
        for members in families.values():
            # 取 family 内最高分作为代表（同分时优先 parent，因为它上下文更完整）
            def _sort_key(r: dict):
                return (
                    float(r.get("score", r.get("fused_score", 0.0))),
                    1 if str(r.get("chunk_role", "parent")) == "parent" else 0,
                )

            members.sort(key=_sort_key, reverse=True)
            best = dict(members[0])
            if len(members) > 1:
                deduped_count += len(members) - 1

            # 若代表是 child 且能反查到父块，展开 content 为父块全文
            if str(best.get("chunk_role", "parent")) == "child" and best.get("parent_id"):
                parent_payload = parent_payloads.get(str(best["parent_id"]))
                if parent_payload:
                    parent_content = str(parent_payload.get("content", ""))
                    if parent_content:
                        matched_child_id = str(best["chunk_id"])
                        best.update(
                            {
                                "chunk_id": str(
                                    parent_payload.get("chunk_id") or best["parent_id"]
                                ),
                                "doc_id": str(parent_payload.get("doc_id") or best["doc_id"]),
                                "content": parent_content,
                                "doc_name": str(
                                    parent_payload.get("doc_name") or best.get("doc_name", "")
                                ),
                                "section_path": list(parent_payload.get("section_path", [])),
                                "page_no": parent_payload.get("page_no"),
                                "chunk_role": "parent",
                                "parent_id": None,
                                "child_ids": list(parent_payload.get("child_ids", []) or []),
                                "chunk_order": parent_payload.get("chunk_order"),
                                "matched_child_id": matched_child_id,
                            }
                        )
                        best["_expanded_from_child"] = True
                        expanded_count += 1

            # 清理内部标记，不进输出
            best.pop("_expanded_from_child", None)
            expanded_rows.append(best)

        # 保持原有排序（eligible_rows 已是降序，families 遍历也保持插入顺序）
        # 但去重后数量变化，重新按 score 降序保证 top_k 切片正确
        expanded_rows.sort(
            key=lambda r: (
                float(r.get("score", r.get("fused_score", 0.0))),
                float(r.get("keyword_score", 0.0)),
            ),
            reverse=True,
        )
        trace = {
            "expanded": expanded_count,
            "deduped": deduped_count,
            "parent_lookup_failed": failed_lookups,
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