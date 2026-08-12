# 对外部依赖（检索存储 / Embedding / 重排）的抽象接口。
# 只保留确实有多个实现、或测试需要替换的契约：
# - SearchStore：生产用 Elasticsearch，测试注入 FakeStore。
# - EmbeddingModel：生产可用 GLM/OpenAI，本地可用 hash。
# - Reranker：rule 与 cross-encoder 两个实现。
# 纯领域数据结构（ParseResultBlock / ChunkMeta / RetrievedChunk / Citation）
# 只服务于单一阶段，已经下沉到各自阶段的 models.py，不再在这里集中定义。

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# 嵌入向量类型别名
EmbeddingVector = list[float]


class EmbeddingModel(Protocol):
    """Embedding 模型协议，将文本编码为向量。"""
    backend_name: str
    max_input_tokens: int | None

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]: ...


def validate_embedding_vector(vector: Sequence[float]) -> None:
    """校验嵌入向量是否为空。"""
    if not vector:
        raise ValueError("embedding vector cannot be empty")


@dataclass
class VectorRecord:
    """Search storage record; context Parents deliberately have no vector."""
    id: str                    # 记录唯一ID
    doc_id: str                # 所属文档ID
    vector: list[float] | None # Child vector; None for context-only Parent
    payload: dict[str, Any] = field(default_factory=dict)   # 附加元数据


@dataclass
class VectorSearchResult:
    """向量检索结果。"""
    id: str                    # 记录唯一ID
    doc_id: str                # 所属文档ID
    score: float               # 相似度得分
    payload: dict[str, Any] = field(default_factory=dict)   # 附加元数据


class SearchStore(Protocol):
    """向量检索存储协议，支持增删改查。"""

    def delete_by_doc_id(self, doc_id: str) -> None: ...   # 按文档ID删除

    def delete_stale_by_doc_id(self, doc_id: str, keep_ids: list[str]) -> None: ...

    def upsert(self, records: list[VectorRecord]) -> None: ...   # 插入或更新向量记录

    def vector_search(
        self,
        query_vector: list[float],                # 查询向量
        top_k: int,                               # 返回前 K 条结果
        filters: dict[str, Any] | None = None,    # 过滤条件
    ) -> list[VectorSearchResult]: ...            # 向量相似度搜索

    def keyword_search(
        self,
        query: str,                                # 关键词查询
        top_k: int,                               # 返回前 K 条结果
        filters: dict[str, Any] | None = None,    # 过滤条件
    ) -> list[dict]: ...                          # 关键词搜索

    def query_by_ids(self, ids: list[str]) -> list[VectorRecord]: ...   # 按ID批量查询


class ChatModel(Protocol):
    """QA model capabilities required for strict prompt budgeting."""

    model_name: str
    context_limit_tokens: int
    completion_reserve_tokens: int

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int: ...

    def complete(self, messages: Sequence[dict[str, str]]) -> str: ...


class Reranker(Protocol):
    """重排序模型协议。"""
    backend_name: str

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]: ...   # 对候选块重新排序
