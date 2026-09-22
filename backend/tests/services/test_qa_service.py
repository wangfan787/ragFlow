"""问答服务：证据模式、上下文预算、来源窗口、模型接线和真实 usage。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.src.apps.services.qa_service import QAService
from backend.src.apps.services.context_window import ContextWindowBuilder
from backend.src.config.qa_config import QAConfig


def _parent_chunk(text: str, child_snippet: str, child_span: tuple[int, int]) -> Document:
    start, end = child_span
    return Document(
        page_content=text,
        metadata={
            "chunk_id": "parent", "doc_id": "doc", "score": 0.8,
            "vector_score": 0.8, "keyword_score": 0.0, "fused_score": 0.8,
            "rerank_score": None, "section_path": ["章节"], "page_no": None,
            "chunk_role": "parent",
            "matched_child_id": "child", "primary_matched_child_id": "child",
            "matched_children": [{
                "chunk_id": "child", "score": 0.8, "snippet": child_snippet,
                "parent_char_start": start, "parent_char_end": end,
                "source_span": {"start_char": start, "end_char": end, "accuracy": "exact"},
            }],
        },
    )


class FakeDetailedRouter:
    """带请求级 trace 的检索替身：验证 QAService 不再依赖共享 last_trace。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.received_configs = []

    def retrieve_detailed(self, question, retrieval_config=None):
        self.received_configs.append(retrieval_config)
        return list(self.chunks), {
            "request_local": True,
            "retrieval_mode": retrieval_config.retrieval_mode,
        }


class FakeAnswerModel:
    model_name = "fake-qa-model"

    def __init__(self, content="答案 [1]", usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(
            content=self.content,
            usage_metadata=self.usage_metadata,
            response_metadata=None,
        )


def _qa_service(chunk: Document, model: FakeAnswerModel | None = None) -> tuple[QAService, FakeAnswerModel]:
    model = model or FakeAnswerModel()
    service = QAService(model=model)
    service.retriever = FakeDetailedRouter([chunk])
    return service, model


def test_qa_child_only_mode_cites_matched_child_text():
    text = "前文" * 100 + "命中子块内容" + "后文" * 100
    start, end = text.index("命中子块内容"), text.index("命中子块内容") + len("命中子块内容")
    chunk = _parent_chunk(text, "命中子块内容", (start, end))
    service, model = _qa_service(chunk)

    payload = service.query("问题", qa_config={"evidence_mode": "child_only"})

    # 证据只包含命中子块文本，引用指向子块而不是父块
    assert "命中子块内容" in model.messages[1]["content"]
    assert "前文前文" not in model.messages[1]["content"]
    assert payload["citations"][0]["chunk_id"] == "child"
    assert payload["trace"]["config"]["qa_config"]["evidence_mode"] == "child_only"


def test_qa_full_parent_mode_uses_entire_parent_text():
    text = "头部证据" + "中段" * 50 + "尾部证据"
    start = text.index("头部证据")
    chunk = _parent_chunk(text, "命中", (start, start + 2))
    service, model = _qa_service(chunk)

    # 窗口设得很小也不影响 full_parent：初始候选就是整个父块
    payload = service.query("问题", qa_config={"evidence_mode": "full_parent", "evidence_window_tokens": 16})

    assert "头部证据" in model.messages[1]["content"]
    assert "尾部证据" in model.messages[1]["content"]
    assert payload["trace"]["config"]["qa_config"]["evidence_mode"] == "full_parent"


def test_qa_window_mode_default_unchanged_and_request_scoped():
    text = "前文" * 2000 + "真正命中的尾部证据" + "后文" * 200
    start = text.index("真正命中的尾部证据")
    chunk = _parent_chunk(text, "真正命中的尾部证据", (start, start + len("真正命中的尾部证据")))
    service, model = _qa_service(chunk)

    # 同一服务实例连续两次请求使用不同窗口配置：请求级 QAConfig 生效
    service.query("第一问", qa_config={"evidence_window_tokens": 300})
    payload_two = service.query("第二问", qa_config={"evidence_window_tokens": 100000})

    assert payload_two["trace"]["config"]["qa_config"] == {
        "context_top_k": 5, "evidence_mode": "window", "evidence_window_tokens": 100000,
        "query_rewrite_enabled": True,
        "colloquial_normalization_enabled": False,
        "step_back_enabled": False,
    }
    # 两次请求的 trace 都是请求局部对象
    assert payload_two["trace"]["request_local"] is True
    assert payload_two["trace"]["retrieval_mode"] == "hybrid"


def test_qa_usage_records_provider_numbers_when_available():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    model = FakeAnswerModel(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    service, _ = _qa_service(chunk, model)

    payload = service.query("问题")
    usage = payload["trace"]["usage"]
    assert usage == {
        "model": "fake-qa-model",
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "cost": "unavailable",
    }


def test_qa_usage_marks_missing_fields_unavailable_instead_of_zero():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    service, _ = _qa_service(chunk)  # FakeAnswerModel 不带 usage_metadata

    payload = service.query("问题")
    usage = payload["trace"]["usage"]
    # 供应商未返回 usage 时不得用 0 冒充真实消耗
    assert usage["input_tokens"] == "unavailable"
    assert usage["output_tokens"] == "unavailable"
    assert usage["total_tokens"] == "unavailable"
    assert usage["cost"] == "unavailable"


def test_qa_usage_reads_openai_compatible_token_usage():
    chunk = _parent_chunk("普通文本", "普通", (0, 2))
    model = FakeAnswerModel()
    model.invoke = lambda messages: SimpleNamespace(
        content="答案 [1]",
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    service, _ = _qa_service(chunk, model)

    usage = service.query("问题")["trace"]["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10


def _document(content="", **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


class FakeRouter:

    def __init__(self, chunk):
        self.chunk = chunk

    def retrieve_detailed(self, question, retrieval_config=None):
        return [self.chunk], {}


def test_qa_default_path_invokes_langchain_and_reuses_model(monkeypatch):
    from backend.src.apps.services import qa_service

    built = []
    model = FakeAnswerModel(content="answer [1]")
    monkeypatch.setattr(qa_service, "build_chat", lambda kind: built.append(kind) or model)
    service = QAService()
    assert built == []
    assert service._generate_answer("question", []).answer == "answer [1]"
    assert service._generate_answer("second", []).answer == "answer [1]"
    assert built == ["qa"]
    assert "second" in model.messages[1]["content"]


@pytest.mark.parametrize("output", [None, ""])
def test_qa_model_failure_or_empty_output_never_becomes_fake_answer(output):
    from fastapi import HTTPException

    def invoke(messages):
        if output is None:
            raise RuntimeError("provider down")
        return SimpleNamespace(content=output)

    service = QAService(model=SimpleNamespace(invoke=invoke))
    with pytest.raises(HTTPException) as error:
        service._generate_answer("question", [])
    assert error.value.status_code == 502


def test_qa_window_is_anchored_near_matched_child_and_citation_uses_prompt_window():
    prefix = "前文" * 2000
    child_text = "真正命中的尾部证据"
    parent_text = prefix + child_text + "后文" * 200
    start = len(prefix)
    chunk = _document(
        chunk_id="parent", doc_id="doc", content=parent_text, score=0.8,
        vector_score=0.8, keyword_score=0.0, fused_score=0.8, rerank_score=None,
        section_path=["tail"], page_no=None, chunk_role="parent",
        matched_child_id="child", primary_matched_child_id="child",
        matched_children=[{
            "chunk_id": "child", "score": 0.8, "snippet": child_text,
            "parent_char_start": start, "parent_char_end": start + len(child_text),
            "source_span": {"start_char": start, "end_char": start + len(child_text)},
        }],
    )
    model = FakeAnswerModel()
    service = QAService(model=model)
    service.context_limit_tokens = 1200
    service.completion_reserve_tokens = 200
    service.prompt_safety_tokens = 256
    service.retriever = FakeRouter(chunk)
    payload = service.query("尾部是什么", qa_config={"evidence_window_tokens": 200})
    assert child_text in model.messages[1]["content"]
    assert payload["citations"][0]["primary_matched_child_id"] == "child"
    assert child_text in payload["citations"][0]["context_snippet"]
    assert payload["trace"]["qa_budget"]["prompt_used_tokens"] <= 744


def test_qa_builds_separate_windows_for_distant_contributing_children():
    text = "A" * 500 + "first-hit" + "B" * 2000 + "second-hit" + "C" * 500
    first = text.index("first-hit")
    second = text.index("second-hit")
    chunk = _document(
        chunk_id="parent", doc_id="doc", content=text, score=0.8,
        vector_score=0.8, keyword_score=0.0, fused_score=0.8, rerank_score=None,
        section_path=[], page_no=None, chunk_role="parent", matched_child_id="c1",
        primary_matched_child_id="c1", matched_children=[
            {"chunk_id": "c1", "snippet": "first-hit", "parent_char_start": first, "parent_char_end": first + 9},
            {"chunk_id": "c2", "snippet": "second-hit", "parent_char_start": second, "parent_char_end": second + 10},
        ],
    )
    service = QAService(model=FakeAnswerModel())
    windows = service.window_builder.candidates([chunk], max_window_tokens=40)
    assert len(windows) == 2
    assert "first-hit" in windows[0].page_content
    assert "second-hit" in windows[1].page_content


def test_qa_missing_coordinates_falls_back_to_child_without_snippet_guessing():
    text = "TARGET" + "middle" * 500 + "TARGET"
    chunk = _document(
        chunk_id="p", doc_id="d", content=text, score=1.0, vector_score=1.0,
        keyword_score=0.0, fused_score=1.0, rerank_score=None, section_path=[], page_no=None,
        matched_child_id="c", primary_matched_child_id="c",
        matched_children=[{"chunk_id": "c", "snippet": "TARGET"}],
    )
    window = QAService(model=FakeAnswerModel()).window_builder.anchored(chunk, 10)
    assert window.page_content == "TARGET"
    assert window.metadata["prompt_span"] == {"fallback": "matched_child"}


def test_window_keeps_input_unchanged_and_uses_absolute_parent_coordinates():
    text = "前文" * 1000 + "命中证据" + "后文" * 1000
    start = text.index("命中证据")
    original = Document(page_content=text, metadata={
        "chunk_id": "p", "doc_id": "d", "chunk_role": "parent", "score": 0.8,
        "matched_children": [{"chunk_id": "c", "snippet": "命中证据",
                              "parent_char_start": start, "parent_char_end": start + 4}],
        "source_span": {"start_line": 1, "end_line": 20},
    })
    before = deepcopy(original.model_dump())
    builder = ContextWindowBuilder()
    first = builder.anchored(original, 100)
    second = builder.anchored(first, 40)
    assert original.model_dump() == before
    assert "命中证据" in second.page_content
    span = second.metadata["prompt_span"]
    assert text[span["parent_char_start"]:span["parent_char_end"]] == second.page_content
    assert span["parent_char_start"] <= start < span["parent_char_end"]


def test_budget_does_not_hide_corrupt_child_coordinates():
    service = QAService()
    service.context_limit_tokens = 200
    service.completion_reserve_tokens = 10
    service.prompt_safety_tokens = 0
    chunk = Document(page_content="长文本 " * 1000, metadata={
        "doc_id": "d", "chunk_id": "p", "score": 0.9,
        "primary_matched_child_id": "c", "matched_children": [{
            "chunk_id": "c", "snippet": "损坏的坐标",
            "parent_char_start": 100000, "parent_char_end": 100010,
        }],
    })
    with pytest.raises(ValueError, match="outside"):
        service._budgeted_evidence("问题", [chunk], QAConfig(evidence_mode="full_parent"))
