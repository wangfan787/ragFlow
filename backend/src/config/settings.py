from __future__ import annotations

import os
from typing import Any, Callable, TypeVar, overload

from .env import config_value, load_config, load_env

T = TypeVar('T')


class Settings:
    """统一配置读取器：YAML 为默认来源，环境变量可选覆盖。
    
    支持类型转换、缓存、验证和默认值。
    
    Example:
        >>> settings = Settings()
        >>> api_key = settings.text("API_KEY")
        >>> debug = settings.bool("DEBUG", default=False)
        >>> timeout = settings.integer("TIMEOUT", default=30, min_value=1, max_value=120)
    """
    
    def __init__(self, *, auto_load: bool = True, cache: bool = True) -> None:
        """初始化设置
        
        Args:
            auto_load: 是否自动加载 .env 文件
            cache: 是否缓存读取的值（仅在 auto_load=True 时有效）
        """
        if auto_load:
            load_env()
            load_config()
        self._cache: dict[str, Any] = {} if cache else None
        self._cache_enabled = cache
    
    def _get(self, name: str, default: Any = None) -> str | None:
        """获取原始环境变量值（带缓存）"""
        if self._cache_enabled and name in self._cache:
            return self._cache[name]
        
        value = os.getenv(name)
        if value is None:
            value = config_value(name)
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
    
    def optional_integer(self, name: str, *, 
                         min_value: int | None = None,
                         max_value: int | None = None) -> int | None:
        """获取可选的整数值"""
        raw = self._get(name)
        if raw is None or raw == "":
            return None
        
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"invalid integer for {name}: {raw!r}")
        
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
    
    def optional_float(self, name: str, *,
                       min_value: float | None = None,
                       max_value: float | None = None) -> float | None:
        """获取可选的浮点数值"""
        raw = self._get(name)
        if raw is None or raw == "":
            return None
        
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"invalid float for {name}: {raw!r}")
        
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
        
        return value
    
    def bool(self, name: str, default: bool = False) -> bool:
        """获取布尔值
        
        支持: 1/0, true/false, yes/no, on/off (不区分大小写)
        """
        raw = self._get(name)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        
        raw_lower = str(raw).lower()
        if raw_lower in {"1", "true", "yes", "on"}:
            return True
        if raw_lower in {"0", "false", "no", "off"}:
            return False
        
        raise ValueError(f"invalid boolean for {name}: {raw!r}")
    
    def list(self, name: str, default: list[str] | None = None,
             separator: str = ",") -> list[str]:
        """获取列表值"""
        raw = self._get(name)
        if raw is None:
            return default or []
        
        return [item.strip() for item in raw.split(separator) if item.strip()]
    
    def dict(self, name: str, default: dict[str, str] | None = None,
             separator: str = ",", kv_separator: str = "=") -> dict[str, str]:
        """获取字典值（键值对列表）"""
        raw = self._get(name)
        if raw is None:
            return default or {}
        
        result = {}
        for item in raw.split(separator):
            if kv_separator in item:
                key, value = item.split(kv_separator, 1)
                result[key.strip()] = value.strip()
        
        return result
    
    @overload
    def get(self, name: str, default: str = "") -> str: ...
    
    @overload
    def get(self, name: str, default: int) -> int: ...
    
    @overload
    def get(self, name: str, default: float) -> float: ...
    
    @overload
    def get(self, name: str, default: bool) -> bool: ...
    
    def get(self, name: str, default: Any = "") -> Any:
        """通用获取方法（自动推断类型）"""
        if isinstance(default, bool):
            return self.bool(name, default)
        if isinstance(default, int):
            return self.integer(name, default)
        if isinstance(default, float):
            return self.float(name, default)
        return self.text(name, str(default))
    
    def require(self, name: str) -> str:
        """获取必需的环境变量"""
        value = self._get(name)
        if value is None:
            raise ValueError(f"required environment variable not set: {name}")
        return value
    
    def clear_cache(self) -> None:
        """清除缓存"""
        if self._cache_enabled:
            self._cache.clear()


# 全局单例
settings = Settings()
