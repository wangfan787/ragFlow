"""切分配置：控制 parent/child 双粒度分块的目标 token 数和行为开关。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """父子双粒度分块配置。

    核心理念：Small-to-Big 检索策略。
    ┌─────────────┐
    │  Parent      │  ~512 token，按 heading/段落边界切
    │  (完整上下文) │  负责提供完整语义上下文
    └──────┬───────┘
           │ 1:N
    ┌──────▼───────┐
    │  Child       │  ~128 token，按句子边界切
    │  (精准召回)   │  负责被用户 query 命中
    └──────────────┘

    检索流程：用户查询 -> 命中 child chunk -> 沿 parent_id 回溯 parent chunk
    -> 返回完整上下文。父子关系天然覆盖跨块边界内容，不需要传统的文本 overlap 拼接。
    """

    # --- 父块参数 ---
    parent_target_tokens: int = 512   # 父块目标大小（软上限，尽量凑到这个量）
    parent_max_tokens: int = 768      # 父块最大大小（硬上限，超过强制截断）

    # --- 子块参数 ---
    child_target_tokens: int = 128    # 子块目标大小（软上限）
    child_max_tokens: int = 192       # 子块最大大小（硬上限）

    # 嵌入只发送子块正文；UTF-8 字节限制由索引入口另外校验。
    embedding_input_budget: int = 192

    # --- 行为开关 ---
    align_to_boundary: bool = True    # 是否对齐语义边界（heading/段落）
    preserve_code_block: bool = True  # 代码块不拆散，整体保留
    preserve_table_block: bool = True # 表格不拆散，整体保留


def build_chunk_config(raw: ChunkConfig | dict | None = None) -> ChunkConfig:
    """构建分块配置的工厂函数。

    支持三种输入：
    - None       -> 返回默认 ChunkConfig()
    - ChunkConfig -> 直接返回（透传）
    - dict       -> 校验字段白名单后构造 ChunkConfig
    """
    if not raw:
        config = ChunkConfig()
    elif isinstance(raw, ChunkConfig):
        config = raw
    else:
        fields = ChunkConfig.__dataclass_fields__
        unknown = sorted(set(raw) - set(fields))
        if unknown:
            raise ValueError(f"unsupported chunk_config fields: {', '.join(unknown)}")
        config = ChunkConfig(**raw)
    if not (0 < config.child_target_tokens <= config.child_max_tokens):
        raise ValueError("child token budgets must satisfy 0 < target <= max")
    if not (0 < config.parent_target_tokens <= config.parent_max_tokens):
        raise ValueError("parent token budgets must satisfy 0 < target <= max")
    if config.child_max_tokens > config.embedding_input_budget:
        raise ValueError("child_max_tokens cannot exceed embedding_input_budget")
    return config
