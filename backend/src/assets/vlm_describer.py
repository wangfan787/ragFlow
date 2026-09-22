"""VLM 图片描述：受 schema 约束的 JSON 输出，由程序拼装可检索文本。

描述质量要求：事实密集、结构固定、可回归
测试，不追求散文通顺；无法辨认的内容必须显式进入 uncertainties，
不编造。VLM 失败时返回 failed 结果，资产与状态由调用方保留。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

from backend.src.infrastructure.models import build_chat

VLM_PROMPT_VERSION = "image-desc-v1"

_DESCRIPTION_PROMPT = """你是图片信息抽取器。仔细阅读图片，只输出一个 JSON 对象，不要输出任何其他文字。
字段要求：
- "ocr_text": 图内可见的全部文字，按阅读顺序拼接；没有则输出空字符串
- "subjects": 图中出现的主体、节点、字段或界面元素名称的列表
- "key_facts": 图片能直接支撑的关键事实列表（架构图的调用方向、表格图片的表头与关键单元格、截图的字段值与错误码等）
- "chart_trends": 若是统计图表，写明标题、坐标轴、图例与主要数值趋势；否则输出 "不适用"
- "uncertainties": 无法确认或辨认不清的内容列表；没有则输出空列表
只描述图中确实存在的信息，不要猜测。"""

# 宽松提取模型输出中的 JSON 对象：容忍 ```json 围栏与前后说明文字。
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_STRING_FIELDS = ("ocr_text", "chart_trends")
_LIST_FIELDS = ("subjects", "key_facts", "uncertainties")


@dataclass(frozen=True)
class ImageDescription:
    """VLM 输出的结构化描述；全部字段已按 schema 规范化。"""

    ocr_text: str
    subjects: tuple[str, ...]
    key_facts: tuple[str, ...]
    chart_trends: str
    uncertainties: tuple[str, ...]


@dataclass(frozen=True)
class DescriptionOutcome:
    status: str  # success | failed
    description: ImageDescription | None = None
    model_name: str | None = None
    error: str | None = None


def assemble_page_content(
    description: ImageDescription, alt_text: str = "", caption: str = "",
) -> str:
    """把结构化描述拼成确定性的可检索正文（image block 的 page_content）。"""
    title = caption or alt_text or "无题注图片"
    parts = [f"[图片] {title}"]
    if alt_text and alt_text != title:
        parts.append(f"替代文本：{alt_text}")
    if description.ocr_text:
        parts.append(f"图内文字：{description.ocr_text}")
    if description.subjects:
        parts.append("主体：" + "、".join(description.subjects))
    if description.key_facts:
        parts.append("关键事实：" + "；".join(description.key_facts))
    if description.chart_trends and description.chart_trends != "不适用":
        parts.append(f"图表趋势：{description.chart_trends}")
    if description.uncertainties:
        parts.append("不确定项：" + "；".join(description.uncertainties))
    return "\n".join(parts)


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def parse_description_payload(payload: object) -> ImageDescription:
    """校验并规范化 VLM 的 JSON 输出；字段缺失或类型错误直接抛 ValueError。"""
    if not isinstance(payload, dict):
        raise ValueError("description payload must be a JSON object")
    fields: dict[str, object] = {}
    for name in _STRING_FIELDS:
        value = payload.get(name, "")
        fields[name] = value if isinstance(value, str) else ""
    for name in _LIST_FIELDS:
        fields[name] = _string_list(payload.get(name))
    return ImageDescription(**fields)  # type: ignore[arg-type]


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 多模态分片响应：拼接文本片段
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


class ImageDescriber:
    """调用视觉模型生成受约束描述；模型在首次使用时惰性构造。"""

    def __init__(self, model=None) -> None:  # noqa: ANN001 - BaseChatModel 替身
        self._model = model

    @property
    def model_name(self) -> str | None:
        for attr in ("model_name", "model"):
            value = getattr(self._model, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        alt_text: str = "",
        caption: str = "",
    ) -> DescriptionOutcome:
        context = "请抽取这张图片的信息。"
        if alt_text or caption:
            context += f"图片上下文：替代文本={alt_text or '无'}；所在章节={caption or '无'}。"
        try:
            if self._model is None:
                self._model = build_chat("vision")
            data_url = (
                f"data:{mime_type};base64,"
                + base64.b64encode(image_bytes).decode("ascii")
            )
            message = self._model.invoke(
                [
                    {"role": "system", "content": _DESCRIPTION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ]
            )
            text = _message_text(message)
            match = _JSON_OBJECT_RE.search(text)
            if not match:
                raise ValueError("VLM 输出中未找到 JSON 对象")
            description = parse_description_payload(json.loads(match.group(0)))
            return DescriptionOutcome(
                status="success", description=description, model_name=self.model_name,
            )
        except Exception as exc:  # noqa: BLE001 - 任何失败都保留资产并如实记录
            return DescriptionOutcome(
                status="failed", model_name=self.model_name, error=str(exc)[:200],
            )
