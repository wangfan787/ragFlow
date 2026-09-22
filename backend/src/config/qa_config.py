"""问答请求配置；默认值统一在 config/defaults.yaml。"""

from typing import ClassVar, Literal
from pydantic import Field, field_validator

from .request_config import RequestConfig, build_config


class QAConfig(RequestConfig):
    section: ClassVar[str] = "qa"
    context_top_k: int = Field(gt=0)
    evidence_mode: Literal["child_only", "window", "full_parent"]
    evidence_window_tokens: int = Field(gt=0)
    query_rewrite_enabled: bool
    colloquial_normalization_enabled: bool
    step_back_enabled: bool

    @field_validator("evidence_mode", mode="before")
    @classmethod
    def normalize_mode(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


def build_qa_config(overrides: QAConfig | dict | None = None, **kwargs) -> QAConfig:
    return build_config(QAConfig, overrides, kwargs)
