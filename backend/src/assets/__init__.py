"""图片资产层：受控存储、归属登记与 VLM 描述。"""

from backend.src.assets.asset_store import (
    AssetFileStore,
    AssetNotFoundError,
    AssetPathError,
    AssetRegistry,
)
from backend.src.assets.vlm_describer import (
    VLM_PROMPT_VERSION,
    DescriptionOutcome,
    ImageDescriber,
)

__all__ = [
    "AssetFileStore",
    "AssetNotFoundError",
    "AssetPathError",
    "AssetRegistry",
    "DescriptionOutcome",
    "ImageDescriber",
    "VLM_PROMPT_VERSION",
]
