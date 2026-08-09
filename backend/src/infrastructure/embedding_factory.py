from __future__ import annotations

import logging

from backend.src.config.settings import settings
from backend.src.contracts import EmbeddingModel
from backend.src.infrastructure.hash_embedding import HashEmbedding
from backend.src.infrastructure.openai_embedding import OpenAIEmbedding

logger = logging.getLogger("mvp_api")

_OPENAI_COMPAT_BACKENDS = {"glm", "openai", "openai-compatible"}
_AUTO_BACKENDS = {"", "auto", None}


def _max_input_tokens(backend: str, model: str) -> int | None:
    if backend != "glm":
        return None
    return {
        "embedding-2": 512,
        "embedding-3": 3072,
    }.get(model.lower())


def _auto_backend() -> str:
    explicit = settings.text("MVP_EMBEDDING_BACKEND").lower()
    if explicit:
        return explicit
    if settings.text("MVP_EMBEDDING_API_KEY"):
        return "glm"
    if settings.text("MVP_EMBEDDING_MODEL"):
        return "glm"
    return "hash"


def build_embedding_model(backend: str | None = None) -> EmbeddingModel:
    normalized = (backend or "").strip().lower()
    if normalized in _AUTO_BACKENDS:
        normalized = _auto_backend()

    if normalized == "hash":
        logger.info("embedding.backend_selected backend=hash dim=64")
        return HashEmbedding()

    if normalized not in _OPENAI_COMPAT_BACKENDS:
        raise ValueError(f"unsupported embedding backend: {normalized}")

    api_key = settings.text("MVP_EMBEDDING_API_KEY")
    model = settings.first("MVP_EMBEDDING_MODEL", default="embedding-3")
    base_url = settings.first(
        "MVP_EMBEDDING_BASE_URL",
        "MVP_METADATA_LLM_BASE_URL",
        default="https://open.bigmodel.cn/api/paas/v4",
    )
    if not api_key:
        raise ValueError("MVP_EMBEDDING_API_KEY is required for the embedding provider")
    if not model:
        raise ValueError("MVP_EMBEDDING_MODEL is required")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for API embeddings") from exc

    dimensions = settings.optional_integer("MVP_EMBEDDING_DIMENSIONS")
    batch_size = settings.integer("MVP_EMBEDDING_BATCH_SIZE", 16, positive=True)
    max_input_tokens = _max_input_tokens(normalized, model)
    max_batch_size = 16 if normalized == "glm" else None
    logger.info(
        "embedding.backend_selected backend=%s model=%s base_url=%s dimensions=%s batch_size=%d max_input_tokens=%s",
        normalized,
        model,
        base_url,
        dimensions,
        batch_size,
        max_input_tokens,
    )
    return OpenAIEmbedding(
        client=OpenAI(api_key=api_key, base_url=base_url),
        model=model,
        backend_name=normalized,
        dimensions=dimensions,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_batch_size=max_batch_size,
    )
