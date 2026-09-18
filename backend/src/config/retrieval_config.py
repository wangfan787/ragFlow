from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RetrievalConfig:
    """检索配置类"""
    top_k: int = 5
    candidate_top_k: int = 30
    similarity_threshold: float = 0.1
    vector_weight: float = 0.75
    keyword_weight: float = 0.25
    rerank_enabled: bool = False
    rerank_backend: str = "rule"
    filters: Optional[dict[str, Any]] = None
    # 检索模式：vector 只走向量通道，keyword 只走 BM25，hybrid 两路召回后融合。
    # 严格消融必须靠该字段让未选中的通道完全不执行，而不是把权重设为 0。
    retrieval_mode: str = "hybrid"
    # 送入 Reranker 的候选数（重排漏斗中段）。None 表示重排整个融合池，
    # 即 candidate_top_k —— 保持旧版本行为，避免存量配置被迫显式声明。
    rerank_top_n: Optional[int] = None

    def __post_init__(self):
        """验证配置有效性"""
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be > 0")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("weights must be >= 0")
        if self.vector_weight + self.keyword_weight <= 0:
            raise ValueError("sum of weights must be > 0")
        if self.rerank_backend not in {"rule", "cross-encoder"}:
            raise ValueError(f"unsupported rerank_backend: {self.rerank_backend}")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval_mode: {self.retrieval_mode}")
        # 重排漏斗必须满足 top_k <= rerank_top_n <= candidate_top_k：
        # 最终结果只能来自送排池，送排池只能来自召回池。
        if self.rerank_top_n is not None and not (
            self.top_k <= self.rerank_top_n <= self.candidate_top_k
        ):
            raise ValueError(
                "rerank_top_n must satisfy "
                f"top_k({self.top_k}) <= rerank_top_n({self.rerank_top_n}) "
                f"<= candidate_top_k({self.candidate_top_k})"
            )


# 使用配置常量
RETRIEVAL_CONFIG_WHITELIST = frozenset({
    "top_k", "candidate_top_k", "similarity_threshold",
    "vector_weight", "keyword_weight", "rerank_enabled",
    "rerank_backend", "filters", "retrieval_mode", "rerank_top_n"
})

# 支持的检索模式：单通道模式用于严格的消融实验（未选中的通道不得执行）
RETRIEVAL_MODES = frozenset({"vector", "keyword", "hybrid"})

RERANK_BACKENDS = frozenset({"rule", "cross-encoder"})


def _as_bool(value: Any) -> bool:
    """安全转换为布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_weights(vector_weight: float, keyword_weight: float) -> tuple[float, float]:
    """归一化权重，确保和为1"""
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("vector_weight and keyword_weight must be >= 0")
    total = vector_weight + keyword_weight
    if total <= 0:
        raise ValueError("vector_weight + keyword_weight must be > 0")
    return (vector_weight / total, keyword_weight / total)


def build_retrieval_config(
    overrides: Optional[RetrievalConfig | dict[str, Any]] = None,
    **kwargs
) -> RetrievalConfig:
    """
    构建检索配置
    
    Args:
        overrides: 覆盖配置（字典或RetrievalConfig对象）
        **kwargs: 额外的键值对覆盖
    
    Examples:
        >>> config = build_retrieval_config(top_k=10, rerank_enabled=True)
        >>> config = build_retrieval_config({"top_k": 10, "filters": {"type": "doc"}})
    """
    # 合并 overrides 和 kwargs
    if overrides is None:
        overrides = {}
    elif isinstance(overrides, RetrievalConfig):
        overrides = asdict(overrides)
    
    # kwargs 优先级更高
    if kwargs:
        overrides = {**overrides, **kwargs}
    
    if not overrides:
        return RetrievalConfig()
    
    # 验证字段白名单
    unknown = sorted(set(overrides.keys()) - RETRIEVAL_CONFIG_WHITELIST)
    if unknown:
        raise ValueError(f"unsupported retrieval_config fields: {', '.join(unknown)}")
    
    data = asdict(RetrievalConfig())
    
    # 字段映射与转换
    field_converters = {
        "top_k": lambda v: int(v),
        "candidate_top_k": lambda v: int(v),
        "similarity_threshold": lambda v: float(v),
        "vector_weight": lambda v: float(v),
        "keyword_weight": lambda v: float(v),
        "rerank_enabled": _as_bool,
        "rerank_backend": lambda v: str(v).strip().lower(),
        "filters": lambda v: dict(v) if isinstance(v, dict) else v,
        "retrieval_mode": lambda v: str(v).strip().lower(),
        # None 表示回退默认语义（等于 candidate_top_k），在下方统一处理
        "rerank_top_n": lambda v: None if v is None else int(v),
    }
    
    for field, converter in field_converters.items():
        if field in overrides:
            try:
                data[field] = converter(overrides[field])
            except (ValueError, TypeError) as e:
                raise ValueError(f"invalid value for {field}: {overrides[field]}") from e
    
    # 处理 filters / rerank_top_n 的 None 特殊情况（显式传 None 表示回退默认语义）
    if "filters" in overrides and overrides["filters"] is None:
        data["filters"] = None
    if "rerank_top_n" in overrides and (overrides["rerank_top_n"] is None or overrides["rerank_top_n"] == ""):
        data["rerank_top_n"] = None
    
    # 归一化权重
    data["vector_weight"], data["keyword_weight"] = _normalize_weights(
        data["vector_weight"], data["keyword_weight"]
    )
    
    # 验证 rerank_backend
    if data["rerank_backend"] not in RERANK_BACKENDS:
        raise ValueError(f"unsupported rerank_backend: {data['rerank_backend']}")
    
    return RetrievalConfig(**data)


# 便利函数
def create_config_from_env(prefix: str = "RETRIEVAL_") -> RetrievalConfig:
    """从环境变量创建配置"""
    import os
    
    config_dict = {}
    for field in RETRIEVAL_CONFIG_WHITELIST:
        env_var = f"{prefix}{field.upper()}"
        value = os.getenv(env_var)
        if value is not None:
            config_dict[field] = value
    
    return build_retrieval_config(config_dict)
