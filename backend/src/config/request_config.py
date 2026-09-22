"""YAML 默认值、严格请求校验、已校验对象直接透传。"""

from typing import ClassVar, TypeVar
from pydantic import BaseModel, ConfigDict

from .settings import settings


class RequestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)
    section: ClassVar[str]

    def __init__(self, **overrides):
        super().__init__(**{**settings.get(self.section), **overrides})


Config = TypeVar("Config", bound=RequestConfig)


def build_config(cls: type[Config], overrides: Config | dict | None, kwargs: dict) -> Config:
    if isinstance(overrides, cls):
        if not kwargs:
            return overrides
        overrides = overrides.model_dump()
    return cls(**{**({} if overrides is None else overrides), **kwargs})
