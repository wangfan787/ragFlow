"""跨请求的会话、预算、压缩与提交边界；模型与检索均为本地替身。"""

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.src.apps.services.common_service import ServiceError
from backend.src.apps.services.conversation_context import working_turns
from backend.src.apps.services.qa_service import QAService
from backend.src.chunking.token_counter import count_message_tokens
from backend.src.config.settings import settings


class Model:
    model_name = "fake"

    def __init__(self, text="The project is Alpha.", reason="stop", error=False):
        self.text, self.reason, self.error = text, reason, error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise RuntimeError("offline")
        return SimpleNamespace(content=self.text, response_metadata={"finish_reason": self.reason},
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                            "output_token_details": {"reasoning": 12}})

    def stream(self, messages):
        message = self.invoke(messages)
        yield SimpleNamespace(content=self.text[:5], response_metadata={}, usage_metadata=None)
        yield SimpleNamespace(content=self.text[5:], response_metadata=message.response_metadata,
                              usage_metadata=message.usage_metadata)


def seed(store, count=1, *, owner="local", large=False):
    session_id = None
    for i in range(count):
        request = store.begin(owner, session_id, f"seed-{i}", {"seed": i})
        question = f"old-marker-{i} " + ("question " * 80 if large else "project Alpha")
        result = store.complete(request, question, {"status": "answered", "citations": [],
            "answer": "fact " * 80 if large else "answer Alpha"})
        session_id = result["session_id"]
    return session_id


def compact_service(monkeypatch, summary=None, model=None):
    monkeypatch.setitem(settings._data["conversation"], "recent_tokens", 50)
    monkeypatch.setitem(settings._data["conversation"], "summary_tokens", 80)
    monkeypatch.setitem(settings._data["conversation"], "summary_output_tokens", 100)
    service = QAService(model=model or Model(), summary_model=summary or Model())
    service.context_limit_tokens = 800
    service.completion_reserve_tokens = 100
    service.prompt_safety_tokens = 20
    return service


@pytest.mark.parametrize("streaming", [False, True])
def test_generation_uses_full_history_and_original_question(streaming):
    model = Model()
    service = QAService(model=model)
    session_id = seed(service.sessions)
    kwargs = {"session_id": session_id, "qa_config": {"answer_mode": "conversation"}, "request_id": "next"}
    if streaming:
        result = list(service.query_stream("翻译上面的回答", **kwargs))[-1][1]
    else:
        result = service.query("翻译上面的回答", **kwargs)
    assert result["status"] == "answered" and result["citations"] == []
    assert result["trace"]["usage"]["reasoning_tokens"] == 12
    assert model.calls[0][1]["content"] == "old-marker-0 project Alpha"
    assert model.calls[0][2]["content"] == "answer Alpha"
    assert "翻译上面的回答" in model.calls[0][-1]["content"]
    assert result["trace"]["qa_budget"]["prompt_used_tokens"] == count_message_tokens(model.calls[0])
    assert service.query("翻译上面的回答", **kwargs) == result
    assert len(model.calls) == 1


def test_sync_and_stream_receive_identical_messages():
    a, b = Model(), Model()
    sync, stream = QAService(model=a), QAService(model=b)
    session_id = seed(sync.sessions)
    kwargs = {"session_id": session_id, "qa_config": {"answer_mode": "conversation"}}
    # 流先暂停在首块，同步从同一快照生成，之后流提交必须发生冲突。
    events = stream.query_stream("总结", **kwargs)
    assert next(events)[0] == "answer"
    sync.query("总结", **kwargs)
    tail = list(events)
    assert a.calls == b.calls
    assert tail[-1][0] == "error" and tail[-1][1]["code"] == "SESSION_CONFLICT"


def test_compression_replaces_history_and_survives_generation_failure(monkeypatch):
    summary = Model("Remember Alpha and its constraints.")
    service = compact_service(monkeypatch, summary, Model(error=True))
    session_id = seed(service.sessions, 6, large=True)
    with pytest.raises(HTTPException):
        service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    saved = service.sessions.read("local", session_id)
    assert saved["summarized_through"] == 5 and len(saved["turns"]) == 6
    assert saved["summary"] == summary.text
    calls = len(summary.calls)
    assert 1 < calls <= settings.integer("conversation.max_summary_calls")
    service.model = Model()
    result = service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    assert len(summary.calls) == calls
    sent = str(service.model.calls[0])
    assert "old-marker-0" not in sent and "old-marker-5" in sent and summary.text in sent
    assert result["trace"]["qa_budget"]["prompt_used_tokens"] < service._prompt_limit()


def test_subsequent_compaction_only_reads_summary_and_uncovered_turns(monkeypatch):
    summary = Model()
    service = compact_service(monkeypatch, summary)
    session_id = seed(service.sessions, 6, large=True)
    service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    summary.calls.clear()
    for i in range(4):
        request = service.sessions.begin("local", session_id, f"extra-{i}", {})
        service.sessions.complete(request, f"new-marker-{i}", {"status": "answered", "answer": "new fact " * 100, "citations": []})
    service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    sent = str(summary.calls)
    assert "old-marker-0" not in sent
    assert "old-marker-5" in sent and "new-marker-0" in sent and summary.text in sent


@pytest.mark.parametrize("bad_summary", [Model(reason="length"), Model(error=True)])
def test_bad_summary_does_not_commit_or_repeat(monkeypatch, bad_summary):
    service = compact_service(monkeypatch, bad_summary)
    session_id = seed(service.sessions, 6, large=True)
    base = service._evidence_messages("继续", [], mode="conversation")
    request = service.sessions.begin("local", session_id, "try", {})
    messages, trace = service.context.prepare(base, request, service.sessions, service._prompt_limit(), 800, 20, service._extract_usage)
    assert trace["status"] == "failed" and trace["dropped_turns"]
    saved = service.sessions.read("local", session_id)
    assert saved["summary"] == "" and saved["summarized_through"] == 0
    assert "old-marker-5" in str(messages) and count_message_tokens(messages) < service._prompt_limit()
    calls = len(bad_summary.calls)
    service.sessions.abandon(request)
    retry = service.sessions.begin("local", session_id, "retry", {})
    _, trace = service.context.prepare(base, retry, service.sessions, service._prompt_limit(), 800, 20, service._extract_usage)
    assert trace["status"] == "skipped_previous_attempt"
    assert len(bad_summary.calls) == calls


def test_no_compaction_below_threshold_even_history_exceeds_recent_target(monkeypatch):
    summary = Model(error=True)
    service = compact_service(monkeypatch, summary)
    session_id = seed(service.sessions, 2, large=True)
    result = service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    assert result["trace"]["compaction"]["status"] == "not_needed"
    assert "old-marker-0" in str(service.model.calls)
    assert summary.calls == []


def test_newest_turn_is_not_silently_discarded(monkeypatch):
    service = compact_service(monkeypatch)
    session_id = seed(service.sessions, large=True)
    service.context_limit_tokens = 220
    with pytest.raises(ServiceError) as exc:
        service.query("继续", session_id=session_id, qa_config={"answer_mode": "conversation"})
    assert exc.value.code == "QA_CONTEXT_BUDGET_EXHAUSTED"
    assert not service.model.calls


@pytest.mark.parametrize("streaming", [False, True])
def test_truncated_answers_are_archived_but_not_working_history(streaming):
    service = QAService(model=Model(reason="length"))
    session_id = seed(service.sessions)
    kwargs = {"session_id": session_id, "qa_config": {"answer_mode": "conversation"}}
    result = list(service.query_stream("hello", **kwargs))[-1][1] if streaming else service.query("hello", **kwargs)
    assert result["status"] == "incomplete"
    session = service.sessions.read("local", result["session_id"])
    assert len(session["turns"]) == 2 and len(working_turns(session)) == 1
    assert result["trace"]["finish_reason"] == "length"


def test_close_or_cancel_stream_does_not_save_half_turn():
    service = QAService(model=Model())
    session_id = seed(service.sessions)
    events = service.query_stream("hello", session_id=session_id, request_id="first", qa_config={"answer_mode": "conversation"})
    assert next(events)[0] == "answer"
    events.close()
    assert len(service.sessions.read("local", session_id)["turns"]) == 1
    cancelled = threading.Event()
    events = service.query_stream("hello", session_id=session_id, request_id="first", qa_config={"answer_mode": "conversation"}, cancelled=cancelled)
    assert next(events)[0] == "answer"
    cancelled.set()
    assert not any(event == "done" for event, _ in events)
    assert len(service.sessions.read("local", session_id)["turns"]) == 1


def test_new_evidence_is_part_of_compaction_trigger(monkeypatch):
    service = compact_service(monkeypatch)
    session_id = seed(service.sessions, 3, large=True)
    request = service.sessions.begin("local", session_id, "new", {})
    base = service._evidence_messages("继续", [], mode="conversation")
    original = service.context.compose(base, "", working_turns(request.session))
    prompt_limit = count_message_tokens(original) + 1
    _, trace = service.context.prepare(base, request, service.sessions, prompt_limit, 2000, 20, service._extract_usage)
    assert trace["status"] == "not_needed"
    base[-1]["content"] += "\n当前证据：" + "evidence " * 100
    _, trace = service.context.prepare(base, request, service.sessions, prompt_limit, 2000, 20, service._extract_usage)
    assert trace["status"] == "compacted"
