from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    import yaml
except ImportError:  # pragma: no cover - reported when YAML config is used
    yaml = None

logger = logging.getLogger("mvp_api")

_LOADED_SIGNATURE: tuple[str, bool] | None = None
_CONFIG_CACHE: tuple[str, dict[str, Any]] | None = None


def _default_env_path() -> Path:
    """获取默认的 .env 文件路径（项目根目录）"""
    return Path(__file__).resolve().parents[3] / ".env"


def _default_config_path() -> Path:
    """Return the backend YAML config path."""
    return Path(__file__).resolve().parents[2] / "config.yaml"


def resolve_config_path() -> Path:
    explicit = os.getenv("MVP_CONFIG_FILE", "").strip()
    return Path(explicit).expanduser() if explicit else _default_config_path()


def load_config(*, force: bool = False) -> dict[str, Any]:
    """Load the YAML config without overriding explicitly exported env vars."""
    global _CONFIG_CACHE

    config_path = resolve_config_path()
    signature = str(config_path)
    if _CONFIG_CACHE and _CONFIG_CACHE[0] == signature and not force:
        return _CONFIG_CACHE[1]

    if not config_path.exists():
        _CONFIG_CACHE = (signature, {})
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the YAML config")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config root must be a mapping: {config_path}")

    _CONFIG_CACHE = (signature, data)
    logger.info("config.yaml_loaded path=%s", config_path)
    return data


_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
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
    "MVP_QA_LLM_THINKING": ("llm", "qa", "thinking"),
    "MVP_METADATA_LLM_API_KEY": ("llm", "metadata", "api_key"),
    "MVP_METADATA_LLM_MODEL": ("llm", "metadata", "model"),
    "MVP_METADATA_LLM_BASE_URL": ("llm", "metadata", "base_url"),
    "MVP_METADATA_LLM_MAX_TOKENS": ("llm", "metadata", "max_tokens"),
    "MVP_METADATA_LLM_THINKING": ("llm", "metadata", "thinking"),
    "MVP_ELASTICSEARCH_URL": ("elasticsearch", "url"),
    "MVP_ELASTICSEARCH_INDEX": ("elasticsearch", "index"),
    "MVP_ELASTICSEARCH_TIMEOUT": ("elasticsearch", "timeout"),
    "MVP_QA_CONTEXT_TOP_K": ("qa", "context_top_k"),
    "MVP_QA_PROMPT_SAFETY_TOKENS": ("qa", "prompt_safety_tokens"),
    "MVP_QA_EVIDENCE_WINDOW_TOKENS": ("qa", "evidence_window_tokens"),
    "MVP_QUERY_REWRITE_ENABLED": ("qa", "query_rewrite", "enabled"),
    "MVP_QUERY_REWRITE_MAX_HISTORY_TURNS": ("qa", "query_rewrite", "max_history_turns"),
    "MVP_QUERY_REWRITE_MAX_HISTORY_TOKENS": ("qa", "query_rewrite", "max_history_tokens"),
    "MVP_QUERY_REWRITE_PROMPT_SAFETY_TOKENS": ("qa", "query_rewrite", "prompt_safety_tokens"),
    "MVP_QUERY_REWRITE_MAX_QUERY_TOKENS": ("qa", "query_rewrite", "max_query_tokens"),
}


def config_value(name: str) -> Any:
    value: Any = load_config()
    for key in _CONFIG_PATHS.get(name, ()):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def resolve_env_path() -> Path:
    """解析环境变量文件路径"""
    explicit = os.getenv("MVP_ENV_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _default_env_path()


def load_env(*, override: bool = False, force: bool = False) -> Optional[Path]:
    """
    加载环境变量文件
    
    Args:
        override: 是否覆盖已存在的环境变量
        force: 是否强制重新加载（忽略缓存）
    
    Returns:
        加载成功的 .env 文件路径，若文件不存在或加载失败则返回 None
    """
    global _LOADED_SIGNATURE

    env_path = resolve_env_path()
    signature = (str(env_path), override)
    
    # 缓存命中且未强制重新加载
    if _LOADED_SIGNATURE == signature and not force:
        return env_path if env_path.exists() else None

    _LOADED_SIGNATURE = signature
    
    # 文件不存在
    if not env_path.exists():
        if os.getenv("MVP_ENV_FILE"):
            logger.warning("config.env_missing path=%s", env_path)
        return None

    # 加载环境变量
    try:
        loaded = load_dotenv(env_path, override=override)
        if loaded:
            logger.info("config.env_loaded path=%s override=%s", env_path, override)
            return env_path
        else:
            logger.warning("config.env_load_failed path=%s", env_path)
            return None
    except Exception as e:
        logger.error("config.env_load_error path=%s error=%s", env_path, e)
        return None
