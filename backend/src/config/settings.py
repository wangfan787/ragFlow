from __future__ import annotations

import os
from typing import Any

from .env import config_value, load_config, load_env

# 没有环境变量和 YAML 值时使用的首版基线；密钥没有默认值。
_DEFAULTS: dict[str, Any] = {
    "MVP_DATA_DIR": "data/md-rag",
    "MVP_EMBEDDING_MODEL": "embedding-3",
    "MVP_EMBEDDING_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
    "MVP_EMBEDDING_DIMENSIONS": 1024,
    "MVP_EMBEDDING_BATCH_SIZE": 16,
    "MVP_QA_LLM_MODEL": "GLM-5.1",
    "MVP_QA_LLM_BASE_URL": "https://open.bigmodel.cn/api/coding/paas/v4",
    "MVP_QA_LLM_CONTEXT_TOKENS": 32768,
    "MVP_QA_COMPLETION_RESERVE_TOKENS": 1024,
    "MVP_QA_LLM_TEMPERATURE": 0.2,
    "MVP_VISION_LLM_MODEL": "glm-4.6v-flash",
    "MVP_VISION_LLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
    "MVP_VISION_LLM_MAX_TOKENS": 512,
    "MVP_VISION_LLM_TEMPERATURE": 0.0,
    "MVP_ELASTICSEARCH_URL": "http://localhost:9200",
    "MVP_ELASTICSEARCH_INDEX": "rag-md-v1",
    "MVP_ELASTICSEARCH_TIMEOUT": 30,
    "MVP_QA_CONTEXT_TOP_K": 5,
    "MVP_QA_EVIDENCE_WINDOW_TOKENS": 384,
    "MVP_QA_PROMPT_SAFETY_TOKENS": 256,
}


class Settings:
    """进程环境 > 根 .env > YAML > 默认值；只提供现有调用方使用的读取方法。"""
    
    def __init__(self, *, auto_load: bool = True, cache: bool = True) -> None:
        if auto_load:
            load_env()
            load_config()
        self._cache: dict[str, Any] = {}
        self._cache_enabled = cache
    
    def _get(self, name: str) -> Any:
        """获取原始环境变量值（带缓存）"""
        if self._cache_enabled and name in self._cache:
            return self._cache[name]
        
        value = os.getenv(name)
        if value is None:
            value = config_value(name)
        if value is None:
            value = _DEFAULTS.get(name)
        if isinstance(value, str):
            value = value.strip()
        
        if self._cache_enabled:
            self._cache[name] = value
        
        return value
    
    def text(self, name: str, default: str = "") -> str:
        """获取字符串值"""
        value = self._get(name)
        return str(value) if value is not None else default
    
    def first(self, *names: str, default: str = "") -> str:
        """从多个键中获取第一个非空值"""
        for name in names:
            value = self.text(name)
            if value:
                return value
        return default
    
    def integer(self, name: str, default: int, *, 
                min_value: int | None = None,
                max_value: int | None = None,
                positive: bool = False) -> int:
        """获取整数值并验证"""
        raw = self._get(name)
        if raw is None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"invalid integer for {name}: {raw!r}")
        
        if positive and value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
        
        return value
    
    def float(self, name: str, default: float, *,
              min_value: float | None = None,
              max_value: float | None = None) -> float:
        """获取浮点数值并验证"""
        raw = self._get(name)
        if raw is None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"invalid float for {name}: {raw!r}")
        
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
        
        return value
    
    def clear_cache(self) -> None:
        """清除缓存"""
        if self._cache_enabled:
            self._cache.clear()


# 全局单例
settings = Settings()
