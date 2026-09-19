"""生产链路评测 Runner：只编排与录制，不复制 backend 算法（评测计划 §4.1）。

模块结构：
- production_adapter  组装生产组件，并把生产 payload 映射为冻结 run schema
- index_dataset       调用生产 IngestionPipeline 写独立评测索引 + index manifest
- run_retrieval       调用生产 HybridRouter（不调 Chat）录制检索 run
- run_qa              调用生产 QAService 录制完整问答 run
- sweep               只在 Dev 上做单变量参数扫描并锁定配置
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

RUNNER_VERSION = "runner-v1"
ROOT = Path(__file__).resolve().parents[2]


def git_commit(repo: Path | None = None) -> str:
    """当前 Git commit；不可得时显式记 unavailable，不伪造。"""
    result = subprocess.run(
        ["git", "-C", str(repo or ROOT), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def config_fingerprint(config: dict) -> str:
    """配置的规范化指纹；run 文件据此证明各题使用的参数一致。"""
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def evaluation_dirs(root: Path | None = None) -> dict[str, Path]:
    """评测产物的标准目录（runs/scores/reports/indexes/locks）。"""
    base = (root or ROOT) / "evaluation"
    return {
        "runs": base / "runs",
        "scores": base / "scores",
        "reports": base / "reports" / "generated",
        "indexes": base / "indexes",
        "locks": base / "locks",
    }
