from __future__ import annotations

import logging

from backend.src.config.retrieval_config import RetrievalConfig, build_retrieval_config
from langchain_core.embeddings import Embeddings

from backend.src.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.infrastructure.rule_reranker import RuleReranker
from backend.src.retrieval.embedding_retriever import EmbeddingRetriever
from backend.src.retrieval.hybrid_fusion import HybridFusion
from backend.src.retrieval.keyword_retriever import KeywordRetriever
from langchain_core.documents import Document
from backend.src.retrieval.vectorizer import query_terms

logger = logging.getLogger("mvp_api")


class HybridRouter:
    """Hybrid router orchestrating keyword + vector + optional rerank."""

    def __init__(
        self,
        store: ElasticsearchStore | None = None,
        embedding_model: Embeddings | None = None,
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

    def _reranker_for_backend(self, backend: str):
        """
        选择相应的reranker
        """
        if backend == "rule":
            return self.reranker
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker()
        return self._cross_encoder

    def _expand_parent_context(
        self,
        eligible_rows: list[dict],
        aggregation: str = "mean",
    ) -> tuple[list[dict], dict]:
        """按 Parent 聚合命中子块，并保留窗口和引用需要的来源信息。"""
        if not eligible_rows:
            return eligible_rows, {
                "expanded": 0, "deduped": 0, "parent_lookup_failed": 0,
                "family_score": aggregation,
            }

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
                payload = {**parent_record.metadata, "content": parent_record.page_content}
                pid = payload["chunk_id"]
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
                key=lambda row: float(row["score"]),
                reverse=True,
            )
            primary = dict(members[0])

            def aggregate(field: str) -> float:
                scores = [float(row.get(field, 0.0) or 0.0) for row in members]
                return max(scores) if aggregation == "max" else sum(scores) / len(scores)

            matched_children = []
            for row in members:
                entry = {
                    "chunk_id": str(row["chunk_id"]),
                    "score": float(row["score"]),
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
                if row.get("asset_id") is not None:  # image 命中子块透传资产引用
                    entry["asset_id"] = str(row["asset_id"])
                matched_children.append(entry)
            family_score = aggregate("score")
            result = {
                **primary,
                "score": family_score,
                "vector_score": aggregate("vector_score"),
                "keyword_score": aggregate("keyword_score"),
                "fused_score": aggregate("fused_score"),
                "rerank_score": (
                    aggregate("rerank_score")
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
                        # image 父块（原子图片）保留资产引用；文本父块该键不存在
                        "asset_id": parent_payload.get("asset_id"),
                    }
                )
                expanded_count += 1
            else:
                # The family score still uses every hit, but only the primary
                # Child text is available to the prompt when Parent lookup fails.
                result["matched_children"] = [matched_children[0]]
            expanded_rows.append(result)

        expanded_rows.sort(
            key=lambda r: (
                float(r["score"]),
                float(r.get("keyword_score", 0.0)),
            ),
            reverse=True,
        )
        trace = {
            "expanded": expanded_count,
            "deduped": len(children) - len(expanded_rows),
            "parent_lookup_failed": failed_lookups,
            "legacy_parent_candidates_dropped": len(eligible_rows) - len(children),
            "family_score": aggregation,
        }
        return expanded_rows, trace

    def retrieve(
        self,
        query: str,
        retrieval_config: RetrievalConfig | dict | None = None,
    ) -> list[Document]:
        """兼容入口：只返回文档；请求级 trace 见 retrieve_detailed()。"""
        results, _trace = self.retrieve_detailed(query, retrieval_config)
        return results

    def retrieve_detailed(
        self,
        query: str,
        retrieval_config: RetrievalConfig | dict | None = None,
    ) -> tuple[list[Document], dict]:
        """执行检索并返回请求级 trace（不依赖共享的 last_trace 状态）。

        通道门控：retrieval_mode 未选中的通道完全不执行，保证
        Vector-only / Keyword-only 是严格消融，而不是权重为 0 的假单通道。
        """
        config = build_retrieval_config(retrieval_config)

        # ---- 通道门控：只执行 retrieval_mode 选中的召回通道 ----
        vector_rows: list[dict] = []
        keyword_rows: list[dict] = []
        if config.retrieval_mode in ("vector", "hybrid"):
            vector_rows = self.embedding_retriever.retrieve(
                query,
                {
                    "top_k": config.candidate_top_k,
                    "filters": config.filters,
                },
            )
        if config.retrieval_mode in ("keyword", "hybrid"):
            keyword_rows = self.keyword_retriever.retrieve(
                query,
                {
                    "top_k": config.candidate_top_k,
                    "filters": config.filters,
                },
            )

        # 单通道模式下把该通道权重置为 1：融合分与通道原始分同尺度，
        # 相似度阈值才能在 vector/keyword/hybrid 三种模式之间横向比较。
        if config.retrieval_mode == "vector":
            vector_weight, keyword_weight = 1.0, 0.0
        elif config.retrieval_mode == "keyword":
            vector_weight, keyword_weight = 0.0, 1.0
        else:
            vector_weight, keyword_weight = config.vector_weight, config.keyword_weight

        fused_rows = self.fusion.fuse(
            vector_rows,
            keyword_rows,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        # 两种粒度共用同一批 Child 候选；Parent 去重不扩大召回池。
        rerank_top_n = (
            config.rerank_top_n
            if config.rerank_top_n is not None
            else config.candidate_top_k
        )
        rerank_pool = fused_rows[:rerank_top_n]
        parent_rerank = config.rerank_enabled and config.rerank_level == "parent"
        rerank_inputs = rerank_pool
        expand_trace = None
        if parent_rerank:
            # 聚合仅准备正文与来源；成功后的最终分数由 Parent CE 覆盖。
            # 此处固定 mean，避免 child_score_aggregation 改变 Parent 模型输入顺序。
            rerank_inputs, expand_trace = self._expand_parent_context(rerank_pool)

        ranked_rows = fused_rows
        fallback_reason: str | None = None
        rerank_model: str | None = None
        rerank_succeeded = False
        rerank_sent_count = 0
        if config.rerank_enabled and rerank_inputs:
            try:
                reranker = self._reranker_for_backend(config.rerank_backend)
                rerank_sent_count = len(rerank_inputs)
                ranked_rows = reranker.rerank(query, rerank_inputs)
                rerank_succeeded = True
                # rule 后端没有模型名；cross-encoder 记录实际模型便于复现
                rerank_model = getattr(reranker, "model_name", None)
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
            if float(item["score"]) >= config.similarity_threshold
        ]

        if parent_rerank and rerank_succeeded:
            # 阈值作用于 Parent CE 分数，不能提前用弱 Child 分数淘汰其父块。
            expand_trace["family_score"] = "cross-encoder"
        else:
            # Child 重排、关闭重排和失败回退共用所选聚合方式。
            eligible_rows, expand_trace = self._expand_parent_context(
                eligible_rows, config.child_score_aggregation,
            )

        results: list[Document] = []
        for item in eligible_rows[: config.top_k]:
            score = float(item["score"])
            fused_score = float(item["fused_score"])
            rerank_score = item.get("rerank_score")
            metadata = {key: value for key, value in item.items() if key != "content"}
            metadata.update(
                score=score, fused_score=fused_score,
                vector_score=float(item.get("vector_score", 0.0)),
                keyword_score=float(item.get("keyword_score", 0.0)),
                rerank_score=float(rerank_score) if rerank_score is not None else None,
            )
            if not metadata.get("doc_id") or not metadata.get("chunk_id"):
                raise ValueError("doc_id and chunk_id are required")
            for field in ("score", "vector_score", "keyword_score", "fused_score", "rerank_score"):
                value = metadata[field]
                if value is not None and not 0.0 <= value <= 1.0:
                    raise ValueError("scores must be between 0 and 1")
            result = Document(page_content=str(item.get("content", "")), metadata=metadata)
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

        trace = {
            "query_terms": sorted(query_terms(query)),
            "chunk_count": len(results),
            "avg_score": (
                round(sum(row.metadata["score"] for row in results) / len(results), 4) if results else 0.0
            ),
            "retrieval_mode": config.retrieval_mode,
            "vector_candidate_count": len(vector_rows),
            "keyword_candidate_count": len(keyword_rows),
            "fused_candidate_count": len(fused_rows),
            "eligible_candidate_count": len(eligible_rows),
            "vector_weight": config.vector_weight,
            "keyword_weight": config.keyword_weight,
            # 单通道模式下的实际生效权重（可能与配置权重不同，见上方说明）
            "vector_weight_effective": vector_weight,
            "keyword_weight_effective": keyword_weight,
            "similarity_threshold": config.similarity_threshold,
            "rerank_enabled": config.rerank_enabled,
            "rerank_backend": config.rerank_backend if config.rerank_enabled else None,
            "rerank_model": rerank_model,
            "rerank_level": config.rerank_level,
            "rerank_level_effective": config.rerank_level if rerank_succeeded else None,
            "child_score_aggregation": config.child_score_aggregation,
            "child_score_aggregation_effective": (
                None if parent_rerank and rerank_succeeded else config.child_score_aggregation
            ),
            # 重排漏斗三段：召回池 -> 送排池 -> 最终 top_k，评测据此复现实验
            "candidate_top_k": config.candidate_top_k,
            "rerank_top_n_effective": rerank_top_n,
            "rerank_child_pool_count": len(rerank_pool) if config.rerank_enabled else 0,
            "rerank_sent_count": rerank_sent_count,
            # Parent 缺失时仅有主命中 Child，仍用同一模型评分并明确记录降级数量。
            "rerank_fallback_child_count": (
                sum(row.get("chunk_role") == "child" for row in rerank_inputs)
                if parent_rerank and rerank_succeeded else 0
            ),
            "final_top_k": config.top_k,
            "fallback_reason": fallback_reason,
            "parent_expansion": expand_trace,
            "chunk_traces": chunk_traces,
            "filtered_chunks": [
                {
                    "doc_id": str(item.get("doc_id", "")),
                    "chunk_id": str(item.get("chunk_id", "")),
                    "score": round(float(item["score"]), 4),
                    "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                    "vector_score": round(float(item.get("vector_score", 0.0)), 4),
                }
                for item in ranked_rows
                if float(item["score"])
                < config.similarity_threshold
            ][:10],
        }
        logger.info(
            "retrieval.hybrid query=%r mode=%s vector_candidates=%d "
            "keyword_candidates=%d fused_candidates=%d "
            "top=%s fallback_reason=%s",
            query,
            config.retrieval_mode,
            len(vector_rows),
            len(keyword_rows),
            len(fused_rows),
            chunk_traces[:5],
            fallback_reason,
        )
        return results, trace
