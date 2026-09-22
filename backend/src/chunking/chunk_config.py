"""父子分块配置；代码/表格保持完整，正文遵守预算。默认值见 YAML。"""

from typing import ClassVar
from pydantic import Field, model_validator

from backend.src.config.request_config import RequestConfig, build_config


class ChunkConfig(RequestConfig):
    section: ClassVar[str] = "chunking"
    parent_target_tokens: int = Field(gt=0)
    parent_max_tokens: int = Field(gt=0)
    child_target_tokens: int = Field(gt=0)
    child_max_tokens: int = Field(gt=0)
    embedding_input_budget: int = Field(gt=0)
    align_to_boundary: bool
    preserve_code_block: bool
    preserve_table_block: bool

    @model_validator(mode="after")
    def validate_budgets(self):
        if self.parent_target_tokens > self.parent_max_tokens:
            raise ValueError("parent target cannot exceed max")
        if not self.child_target_tokens <= self.child_max_tokens <= self.embedding_input_budget:
            raise ValueError("child budgets must satisfy target <= max <= embedding_input_budget")
        return self


def build_chunk_config(raw: ChunkConfig | dict | None = None) -> ChunkConfig:
    return build_config(ChunkConfig, raw, {})
