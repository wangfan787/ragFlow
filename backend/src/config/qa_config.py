"""请求级 QA 配置：证据窗口参数必须随请求传入，而不是只在服务初始化时读全局环境。

背景（docs/plan/README.md §4.2 前置改造 3）：`context_top_k` 和
`evidence_window_tokens` 原来在 `QAService.__init__` 从全局配置读取，
评测 Runner 无法在同一进程内逐题、逐配置做可靠 A/B。本模块提供不可变
的请求级配置对象，`QAService.query()` 每次调用时显式接收。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

# 证据构造模式：
# - child_only:   只把命中子块自身文本作为证据（最小上下文基线）
# - window:       以命中子块为锚点在父块内截取窗口（当前默认行为）
# - full_parent:  直接使用整个父块文本（最大上下文基线）
EVIDENCE_MODES = frozenset({"child_only", "window", "full_parent"})


@dataclass(frozen=True)
class QAConfig:
    """单次问答请求的证据构造配置（不可变，可安全写入 trace）"""

    context_top_k: int = 5
    evidence_mode: str = "window"
    evidence_window_tokens: int = 384

    def __post_init__(self) -> None:
        if self.context_top_k <= 0:
            raise ValueError("context_top_k must be > 0")
        if self.evidence_window_tokens <= 0:
            raise ValueError("evidence_window_tokens must be > 0")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"unsupported evidence_mode: {self.evidence_mode} "
                f"(expected one of {sorted(EVIDENCE_MODES)})"
            )

    def as_trace_dict(self) -> dict[str, Any]:
        """写入 trace 的纯字典快照（与 dataclass 字段一一对应）"""
        return asdict(self)


QA_CONFIG_WHITELIST = frozenset({
    "context_top_k", "evidence_mode", "evidence_window_tokens",
})


def build_qa_config(
    overrides: Optional[QAConfig | dict[str, Any]] = None,
    **kwargs
) -> QAConfig:
    """构建请求级 QA 配置；解析规则与 build_retrieval_config 保持一致。

    Examples:
        >>> build_qa_config(evidence_mode="child_only")
        >>> build_qa_config({"evidence_window_tokens": 192}, context_top_k=3)
    """
    if overrides is None:
        overrides = {}
    elif isinstance(overrides, QAConfig):
        overrides = asdict(overrides)

    # kwargs 优先级更高
    if kwargs:
        overrides = {**overrides, **kwargs}

    if not overrides:
        return QAConfig()

    unknown = sorted(set(overrides.keys()) - QA_CONFIG_WHITELIST)
    if unknown:
        raise ValueError(f"unsupported qa_config fields: {', '.join(unknown)}")

    data = asdict(QAConfig())
    field_converters = {
        "context_top_k": lambda v: int(v),
        "evidence_mode": lambda v: str(v).strip().lower(),
        "evidence_window_tokens": lambda v: int(v),
    }
    for field, converter in field_converters.items():
        if field in overrides:
            if overrides[field] is None:
                raise ValueError(f"qa_config field {field} cannot be None")
            try:
                data[field] = converter(overrides[field])
            except (ValueError, TypeError) as e:
                raise ValueError(f"invalid value for {field}: {overrides[field]}") from e

    return QAConfig(**data)
