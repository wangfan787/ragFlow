
from __future__ import annotations

import re


def chunk_order(row: dict) -> int:
    # 优先读 chunker 写入的显式 chunk_order（父子分块下更稳，不依赖 id 格式）。
    explicit = row.get("chunk_order")
    if isinstance(explicit, int):
        return explicit
    # 回退：从 chunk_id 解析（向后兼容旧数据）。
    match = re.search(r"_ck_(\d+)$", str(row.get("chunk_id", "")))
    return int(match.group(1)) if match else 1_000_000