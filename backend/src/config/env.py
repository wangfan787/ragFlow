"""读取根 .env 和后端 YAML；保留 MVP_* 显式映射。"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

_CONFIG_CACHE: tuple[str, dict[str, Any]] | None = None
_LOADED_SIGNATURE: tuple[str, bool] | None = None


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".env"


def resolve_env_path() -> Path:
    return Path(os.getenv("MVP_ENV_FILE") or _default_env_path()).expanduser()


def resolve_config_path() -> Path:
    default = Path(__file__).resolve().parents[2] / "config.yaml"
    return Path(os.getenv("MVP_CONFIG_FILE") or default).expanduser()


def load_env(*, override: bool = False, force: bool = False) -> Path | None:
    global _LOADED_SIGNATURE
    path = resolve_env_path()
    signature = (str(path), override)
    if not path.is_file():
        return None
    if force or _LOADED_SIGNATURE != signature:
        load_dotenv(path, override=override)
        _LOADED_SIGNATURE = signature
    return path


def load_config(*, force: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    path = resolve_config_path()
    if _CONFIG_CACHE and _CONFIG_CACHE[0] == str(path) and not force:
        return _CONFIG_CACHE[1]
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
    if not isinstance(data, dict):
        raise ValueError("配置文件根节点必须是字典")
    _CONFIG_CACHE = (str(path), data)
    return data

_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "MVP_DATA_DIR": ("data_dir",),
    "MVP_EMBEDDING_BACKEND": ("embedding", "backend"),
    "MVP_EMBEDDING_API_KEY": ("embedding", "api_key"),
    "MVP_EMBEDDING_MODEL": ("embedding", "model"),
    "MVP_EMBEDDING_BASE_URL": ("embedding", "base_url"),
    "MVP_EMBEDDING_DIMENSIONS": ("embedding", "dimensions"),
    "MVP_EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
    "MVP_QA_LLM_API_KEY": ("llm", "qa", "api_key"),
    "MVP_QA_LLM_MODEL": ("llm", "qa", "model"),
    "MVP_QA_LLM_BASE_URL": ("llm", "qa", "base_url"),
    "MVP_QA_COMPLETION_RESERVE_TOKENS": ("llm", "qa", "completion_reserve_tokens"),
    "MVP_QA_LLM_CONTEXT_TOKENS": ("llm", "qa", "context_limit_tokens"),
    "MVP_QA_LLM_TEMPERATURE": ("llm", "qa", "temperature"),
    "MVP_VISION_LLM_API_KEY": ("llm", "vision", "api_key"),
    "MVP_VISION_LLM_MODEL": ("llm", "vision", "model"),
    "MVP_VISION_LLM_BASE_URL": ("llm", "vision", "base_url"),
    "MVP_VISION_LLM_MAX_TOKENS": ("llm", "vision", "max_tokens"),
    "MVP_VISION_LLM_TEMPERATURE": ("llm", "vision", "temperature"),
    "MVP_ELASTICSEARCH_URL": ("elasticsearch", "url"),
    "MVP_ELASTICSEARCH_INDEX": ("elasticsearch", "index"),
    "MVP_ELASTICSEARCH_TIMEOUT": ("elasticsearch", "timeout"),
    "MVP_QA_CONTEXT_TOP_K": ("qa", "context_top_k"),
    "MVP_QA_PROMPT_SAFETY_TOKENS": ("qa", "prompt_safety_tokens"),
    "MVP_QA_EVIDENCE_WINDOW_TOKENS": ("qa", "evidence_window_tokens"),
}


def config_value(name: str) -> Any:
    path = _CONFIG_PATHS.get(name)
    if path is None:
        return None
    value = load_config()
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value
