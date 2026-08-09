from types import SimpleNamespace

import pytest
import tiktoken

from backend.src.infrastructure.embedding_factory import _max_input_tokens
from backend.src.infrastructure.openai_embedding import OpenAIEmbedding


class _FakeEmbeddings:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        data = [
            SimpleNamespace(index=index, embedding=[float(index), 1.0])
            for index in reversed(range(len(request["input"])))
        ]
        return SimpleNamespace(data=data)


def _model(*, batch_size=16, max_input_tokens=3072, max_batch_size=None):
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    model = OpenAIEmbedding(
        client=client,
        model="embedding-3",
        backend_name="glm",
        dimensions=2,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_batch_size=max_batch_size,
    )
    return model, embeddings


def test_glm_model_token_limits_match_ragflow():
    assert _max_input_tokens("glm", "embedding-2") == 512
    assert _max_input_tokens("glm", "embedding-3") == 3072
    assert _max_input_tokens("openai", "embedding-3") is None


def test_truncates_each_input_by_tokens_before_request():
    model, embeddings = _model(max_input_tokens=32)
    long_text = "中英文 mixed text " * 100

    vectors = model.encode(["short", long_text])

    encoder = tiktoken.get_encoding("cl100k_base")
    sent = embeddings.requests[0]["input"]
    assert sent[0] == "short"
    assert len(encoder.encode(sent[1])) <= 32
    assert len(sent[1]) < len(long_text)
    assert vectors == [[0.0, 1.0], [1.0, 1.0]]


def test_batches_requests_without_changing_input_order():
    model, embeddings = _model(batch_size=2)

    vectors = model.encode(["a", "b", "c"])

    assert [request["input"] for request in embeddings.requests] == [["a", "b"], ["c"]]
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert all(request["dimensions"] == 2 for request in embeddings.requests)


def test_provider_batch_limit_cannot_be_overridden_by_configuration():
    model, embeddings = _model(batch_size=64, max_batch_size=16)

    model.encode([f"text-{index}" for index in range(17)])

    assert [len(request["input"]) for request in embeddings.requests] == [16, 1]


def test_reserved_token_text_is_treated_as_plain_input():
    model, embeddings = _model(max_input_tokens=32)

    model.encode(["prefix <|endoftext|> suffix"])

    assert embeddings.requests[0]["input"] == ["prefix <|endoftext|> suffix"]


def test_tokenizer_failure_does_not_submit_unchecked_text():
    model, embeddings = _model()
    model._token_counter = SimpleNamespace(count=lambda text: 0, truncate=lambda text, limit: text)

    with pytest.raises(RuntimeError, match="failed to count tokens"):
        model.encode(["non-empty"])

    assert embeddings.requests == []
