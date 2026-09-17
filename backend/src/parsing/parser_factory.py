# 导入未来特性以支持延迟注解求值
from __future__ import annotations

# 所有格式统一由 unstructured 引擎解析，输出一致的 ParseResultBlock 契约
from .unstructured_parser import UnstructuredParser

_SUPPORTED_TYPES = {"md", "markdown", "pdf", "txt", "text", "html", "htm"}


def build_parser(file_type: str):
    """按文件类型返回解析器实例。

    自 unstructured 统一引擎后，各格式共用同一个 UnstructuredParser；
    这里保留工厂函数是为了维持「file_type 校验 + 统一入口」的调用契约，
    ingestion_pipeline 与评测适配器都不感知引擎细节。
    """
    normalized = (file_type or "").strip().lower().lstrip(".")
    if normalized not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported file_type for parser: {file_type}")
    return UnstructuredParser()
