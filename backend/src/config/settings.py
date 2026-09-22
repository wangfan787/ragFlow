"""只读取 config/defaults.yaml + config/local.yaml，不导入环境变量。"""

from copy import deepcopy
from pathlib import Path
import math

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "defaults.yaml"
LOCAL_CONFIG = REPO_ROOT / "config" / "local.yaml"


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        # 不把可能含密钥的 YAML 原文带进异常消息。
        raise ValueError(f"Invalid YAML configuration: {path}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def _merge(base: dict, overrides: dict, prefix: str = "") -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"Unknown configuration field: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section must be a mapping: {name}")
            result[key] = _merge(base[key], value, f"{name}.")
        else:
            result[key] = value
    return result


class Settings:
    """读取一次；测试可显式传 local_path / overrides，不改变进程环境。"""

    def __init__(self, *, local_path: Path | None = LOCAL_CONFIG, overrides: dict | None = None):
        self._data = _read_yaml(DEFAULT_CONFIG)
        if local_path is not None and local_path.is_file():
            self._data = _merge(self._data, _read_yaml(local_path))
        if overrides is not None:
            self._data = _merge(self._data, overrides)

    def get(self, name: str):
        value = self._data
        for key in name.split("."):
            value = value[key]
        return deepcopy(value)

    def text(self, name: str) -> str:
        value = self.get(name)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value

    def integer(self, name: str, *, min_value: int | None = None, max_value: int | None = None,
                positive: bool = False) -> int:
        value = self.get(name)
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        if (positive and value <= 0) or (min_value is not None and value < min_value):
            raise ValueError(f"{name} is below its minimum")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} exceeds its maximum")
        return value

    def float(self, name: str) -> float:
        value = self.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(value)


settings = Settings()
