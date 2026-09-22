"""检索请求配置；默认值统一在 config/defaults.yaml。"""

from typing import Any, ClassVar, Literal
from pydantic import Field, field_validator, model_validator

from .request_config import RequestConfig, build_config


class RetrievalConfig(RequestConfig):
    section: ClassVar[str] = "retrieval"
    top_k: int = Field(gt=0)
    candidate_top_k: int = Field(gt=0)
    similarity_threshold: float = Field(ge=0, le=1)
    vector_weight: float = Field(ge=0)
    keyword_weight: float = Field(ge=0)
    rerank_enabled: bool
    rerank_backend: Literal["rule", "cross-encoder"]
    filters: dict[str, Any] | None
    retrieval_mode: Literal["vector", "keyword", "hybrid"]
    rerank_top_n: int | None = Field(ge=1)

    @field_validator("retrieval_mode", "rerank_backend", mode="before")
    @classmethod
    def normalize_name(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_relationships(self):
        total = self.vector_weight + self.keyword_weight
        if total <= 0:
            raise ValueError("sum of weights must be > 0")
        object.__setattr__(self, "vector_weight", self.vector_weight / total)
        object.__setattr__(self, "keyword_weight", self.keyword_weight / total)
        if self.rerank_top_n is not None and not self.top_k <= self.rerank_top_n <= self.candidate_top_k:
            raise ValueError("rerank_top_n must satisfy top_k <= rerank_top_n <= candidate_top_k")
        return self


def build_retrieval_config(overrides: RetrievalConfig | dict | None = None, **kwargs) -> RetrievalConfig:
    return build_config(RetrievalConfig, overrides, kwargs)
