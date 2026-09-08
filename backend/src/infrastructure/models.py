"""首版模型入口：只组装 LangChain 组件，不实现 HTTP 客户端或生成回退。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from backend.src.config.settings import Settings, settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


class ModelConfigurationError(ValueError):
    """调用模型前发现缺失或不合法的配置；错误不包含凭据值。"""


def _require(config: Settings, name: str) -> str:
    value = config.text(name)
    if not value:
        raise ModelConfigurationError(f"模型配置缺失：请设置 {name}")
    return value


def build_chat(
    kind: Literal["qa", "vision"] = "qa",
    *,
    config: Settings | None = None,
    chat_factory: Callable[..., BaseChatModel] | None = None,
) -> BaseChatModel:
    """在服务实际调用时构造；测试可注入构造函数，不引入生产 fallback。"""
    if kind not in {"qa", "vision"}:
        raise ValueError("聊天模型类型仅支持 qa 或 vision")
    config = config if config is not None else settings
    prefix = f"MVP_{kind.upper()}_LLM"
    if kind == "vision":
        api_key = config.first(f"{prefix}_API_KEY", "MVP_EMBEDDING_API_KEY")
        if not api_key:
            raise ModelConfigurationError(
                "视觉模型配置缺失：请设置 MVP_VISION_LLM_API_KEY 或 MVP_EMBEDDING_API_KEY"
            )
        max_tokens = config.integer("MVP_VISION_LLM_MAX_TOKENS", 512, positive=True)
    else:
        api_key = _require(config, f"{prefix}_API_KEY")
        max_tokens = config.integer("MVP_QA_COMPLETION_RESERVE_TOKENS", 1024, positive=True)
        context_limit = config.integer("MVP_QA_LLM_CONTEXT_TOKENS", 32768, positive=True)
        safety = config.integer("MVP_QA_PROMPT_SAFETY_TOKENS", 256, min_value=0)
        if max_tokens + safety >= context_limit:
            raise ModelConfigurationError("QA 总上下文必须大于输出预留与安全余量之和")

    options = {
        "model": _require(config, f"{prefix}_MODEL"),
        "base_url": _require(config, f"{prefix}_BASE_URL"),
        "api_key": api_key,
        "temperature": config.float(f"{prefix}_TEMPERATURE", 0.2 if kind == "qa" else 0.0),
        "timeout": 60,
        "max_retries": 1,
        "use_responses_api": False,
        # GLM 接受 max_tokens；避免 LangChain 将构造参数改为 max_completion_tokens。
        "extra_body": {"max_tokens": max_tokens, "thinking": {"type": "disabled"}},
    }
    if chat_factory is None:
        from langchain_openai import ChatOpenAI

        chat_factory = ChatOpenAI
    return chat_factory(**options)


def build_embeddings(
    *,
    config: Settings | None = None,
    embeddings_factory: Callable[..., Embeddings] | None = None,
) -> Embeddings:
    """原始字符串直接交给供应商；长度校验由子块/索引阶段负责。"""
    config = config if config is not None else settings
    options = {
        "api_key": _require(config, "MVP_EMBEDDING_API_KEY"),
        "model": _require(config, "MVP_EMBEDDING_MODEL"),
        "base_url": _require(config, "MVP_EMBEDDING_BASE_URL"),
        "dimensions": config.integer("MVP_EMBEDDING_DIMENSIONS", 1024, positive=True),
        "chunk_size": config.integer("MVP_EMBEDDING_BATCH_SIZE", 16, min_value=1, max_value=16),
        "check_embedding_ctx_length": False,
        "model_kwargs": {"encoding_format": "float"},
        "timeout": 60,
        "max_retries": 1,
    }
    if embeddings_factory is None:
        from langchain_openai import OpenAIEmbeddings

        embeddings_factory = OpenAIEmbeddings
    return embeddings_factory(**options)
