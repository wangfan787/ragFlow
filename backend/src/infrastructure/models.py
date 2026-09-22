"""统一 LangChain 模型入口，配置来自 YAML，缺失密钥直接报错。"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from backend.src.config.settings import Settings, settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


class ModelConfigurationError(ValueError):
    """错误只报告字段名，不包含凭据值。"""


def _require(config: Settings, name: str) -> str:
    value = config.text(name)
    if not value:
        raise ModelConfigurationError(f"模型配置缺失：请在 config/local.yaml 设置 {name}")
    return value


def build_chat(kind: Literal["qa", "vision"] = "qa", *, config: Settings | None = None,
               chat_factory: Callable[..., "BaseChatModel"] | None = None) -> "BaseChatModel":
    if kind not in {"qa", "vision"}:
        raise ValueError("聊天模型类型仅支持 qa 或 vision")
    config = config if config is not None else settings
    prefix = f"llm.{kind}"
    max_tokens = config.integer(
        f"{prefix}.completion_reserve_tokens" if kind == "qa" else f"{prefix}.max_tokens", positive=True,
    )
    if kind == "qa":
        limit = config.integer("llm.qa.context_limit_tokens", positive=True)
        safety = config.integer("llm.qa.prompt_safety_tokens", min_value=0)
        if max_tokens + safety >= limit:
            raise ModelConfigurationError("QA 总上下文必须大于输出预留与安全余量之和")
    options = {
        "model": _require(config, f"{prefix}.model"),
        "base_url": _require(config, f"{prefix}.base_url"),
        "api_key": _require(config, f"{prefix}.api_key"),
        "temperature": config.float(f"{prefix}.temperature"),
        "timeout": config.integer("llm.timeout_seconds", positive=True),
        "max_retries": config.integer("llm.max_retries", min_value=0),
        "use_responses_api": False,
        # GLM 使用 max_tokens；明确关闭 thinking，保持当前 API 契约。
        "extra_body": {"max_tokens": max_tokens, "thinking": {"type": "disabled"}},
    }
    if chat_factory is None:
        from langchain_openai import ChatOpenAI
        chat_factory = ChatOpenAI
    return chat_factory(**options)


def build_embeddings(*, config: Settings | None = None,
                     embeddings_factory: Callable[..., "Embeddings"] | None = None) -> "Embeddings":
    config = config if config is not None else settings
    options = {
        "api_key": _require(config, "embedding.api_key"),
        "model": _require(config, "embedding.model"),
        "base_url": _require(config, "embedding.base_url"),
        "dimensions": config.integer("embedding.dimensions", positive=True),
        "chunk_size": config.integer("embedding.batch_size", min_value=1, max_value=16),
        "check_embedding_ctx_length": False,
        "model_kwargs": {"encoding_format": "float"},
        "timeout": config.integer("llm.timeout_seconds", positive=True),
        "max_retries": config.integer("llm.max_retries", min_value=0),
    }
    if embeddings_factory is None:
        from langchain_openai import OpenAIEmbeddings
        embeddings_factory = OpenAIEmbeddings
    return embeddings_factory(**options)
