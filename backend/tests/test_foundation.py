"""阶段 01：配置、真实适配器的模拟 HTTP 请求、无外部服务启动。

从仓库根目录使用 conda agent 的解释器运行 pytest；不需要模型 key 或 ES。
首次实际 token 计数需要 tiktoken 的 cl100k_base 词表缓存。
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
import yaml

from backend.src.config import env
from backend.src.config.settings import Settings, settings
from backend.src.infrastructure.models import (
    ModelConfigurationError,
    build_chat,
    build_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    # 测试不用用户的 key、.env 或 YAML；dotenv 写入的变量也要在退出时恢复。
    original = {k: v for k, v in os.environ.items() if k.startswith("MVP_")}
    for name in original:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MVP_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("MVP_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.setattr(env, "_CONFIG_CACHE", None)
    monkeypatch.setattr(env, "_LOADED_SIGNATURE", None)
    settings.clear_cache()
    yield config_path
    for name in list(os.environ):
        if name.startswith("MVP_"):
            del os.environ[name]
    os.environ.update(original)
    settings.clear_cache()


@pytest.mark.parametrize(
    "process_value,dotenv_value,yaml_value,expected",
    [
        ("process-model", "dotenv-model", "yaml-model", "process-model"),
        (None, "dotenv-model", "yaml-model", "dotenv-model"),
        (None, None, "yaml-model", "yaml-model"),
        (None, None, None, "embedding-3"),
    ],
)
def test_configuration_priority(
    monkeypatch, isolated_config, process_value, dotenv_value, yaml_value, expected
):
    isolated_config.write_text(
        yaml.safe_dump({"embedding": {"model": yaml_value}}), encoding="utf-8"
    )
    if dotenv_value is not None:
        isolated_config.with_name(".env").write_text(
            f"MVP_EMBEDDING_MODEL={dotenv_value}\n", encoding="utf-8"
        )
    if process_value is not None:
        monkeypatch.setenv("MVP_EMBEDDING_MODEL", process_value)
    assert Settings().text("MVP_EMBEDDING_MODEL") == expected


def test_default_root_env_and_unknown_setting():
    assert env._default_env_path() == REPO_ROOT / ".env"
    assert env.config_value("UNKNOWN_SETTING") is None
    assert Settings().text("UNKNOWN_SETTING", "fallback") == "fallback"
    assert Settings().integer("UNKNOWN_NUMBER", 7) == 7


def test_existing_model_and_es_mappings(isolated_config):
    isolated_config.write_text(yaml.safe_dump({
        "embedding": {"model": "custom-embedding", "dimensions": 256, "batch_size": 3},
        "llm": {
            "qa": {"model": "custom-qa", "base_url": "https://qa.example/v1", "api_key": "qa-key"},
            "metadata": {"model": "legacy", "base_url": "https://legacy.example/v1", "api_key": "old-key"},
            "vision": {"model": "custom-vision", "base_url": "https://vision.example/v1", "api_key": "vision-key"},
        },
        "elasticsearch": {"url": "http://es.example:9200", "index": "existing-index", "timeout": 12},
    }), encoding="utf-8")
    config = Settings()
    assert config.text("MVP_EMBEDDING_MODEL") == "custom-embedding"
    assert config.integer("MVP_EMBEDDING_DIMENSIONS", 1024) == 256
    assert config.integer("MVP_EMBEDDING_BATCH_SIZE", 16) == 3
    qa = build_chat("qa", config=config, chat_factory=lambda **kw: kw)
    assert (qa["model"], qa["base_url"], qa["api_key"]) == (
        "custom-qa", "https://qa.example/v1", "qa-key"
    )
    vision = build_chat("vision", config=config, chat_factory=lambda **kw: kw)
    assert (vision["model"], vision["base_url"], vision["api_key"]) == (
        "custom-vision", "https://vision.example/v1", "vision-key"
    )
    assert config.text("MVP_ELASTICSEARCH_URL") == "http://es.example:9200"
    assert config.text("MVP_ELASTICSEARCH_INDEX") == "existing-index"
    assert config.integer("MVP_ELASTICSEARCH_TIMEOUT", 30) == 12


def test_missing_keys_do_not_use_metadata_or_openai_fallback(monkeypatch):
    monkeypatch.setenv("MVP_METADATA_LLM_API_KEY", "legacy-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")

    def unexpected_factory(**kwargs):
        pytest.fail("缺失配置时不应构造模型")

    config = Settings()
    with pytest.raises(ModelConfigurationError, match="MVP_QA_LLM_API_KEY"):
        build_chat("qa", config=config, chat_factory=unexpected_factory)
    with pytest.raises(ModelConfigurationError, match="MVP_VISION_LLM_API_KEY"):
        build_chat("vision", config=config, chat_factory=unexpected_factory)
    with pytest.raises(ModelConfigurationError, match="MVP_EMBEDDING_API_KEY"):
        build_embeddings(config=config, embeddings_factory=unexpected_factory)


def test_vision_key_fallback_is_only_embedding(monkeypatch):
    monkeypatch.setenv("MVP_EMBEDDING_API_KEY", "embedding-test-key")
    config = Settings(cache=False)
    assert build_chat("vision", config=config, chat_factory=lambda **kw: kw)["api_key"] == "embedding-test-key"
    with pytest.raises(ModelConfigurationError, match="MVP_QA_LLM_API_KEY"):
        build_chat("qa", config=config)
    monkeypatch.setenv("MVP_VISION_LLM_API_KEY", "vision-test-key")
    assert build_chat("vision", config=config, chat_factory=lambda **kw: kw)["api_key"] == "vision-test-key"
    monkeypatch.setenv("MVP_VISION_LLM_API_KEY", "")
    assert build_chat("vision", config=config, chat_factory=lambda **kw: kw)["api_key"] == "embedding-test-key"


def test_invalid_model_configuration_fails_before_construction(monkeypatch):
    monkeypatch.setenv("MVP_QA_LLM_API_KEY", "qa-test-key")
    monkeypatch.setenv("MVP_QA_LLM_CONTEXT_TOKENS", "1280")
    with pytest.raises(ModelConfigurationError, match="总上下文"):
        build_chat("qa", config=Settings())
    with pytest.raises(ValueError, match="qa 或 vision"):
        build_chat("metadata")
    monkeypatch.setenv("MVP_EMBEDDING_API_KEY", "embedding-test-key")
    monkeypatch.setenv("MVP_EMBEDDING_BATCH_SIZE", "17")
    with pytest.raises(ValueError, match="MVP_EMBEDDING_BATCH_SIZE"):
        build_embeddings(config=Settings())


@pytest.fixture
def mock_http():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "id": "chat-test", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "测试回复"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            })
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(200, json={
            "object": "list", "model": body["model"],
            "data": [{"object": "embedding", "index": i, "embedding": [0.25] * 1024} for i, _ in enumerate(body["input"])],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        })

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        async_client = httpx.AsyncClient(transport=transport)
        try:
            yield requests, client, async_client
        finally:
            asyncio.run(async_client.aclose())


@pytest.mark.parametrize("kind,model_name,max_tokens,temperature,path", [
    ("qa", "GLM-5.1", 1024, 0.2, "/api/coding/paas/v4/chat/completions"),
    ("vision", "glm-4.6v-flash", 512, 0, "/api/paas/v4/chat/completions"),
])
def test_chat_wire_request(monkeypatch, mock_http, kind, model_name, max_tokens, temperature, path):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv(f"MVP_{kind.upper()}_LLM_API_KEY", "test-key")
    requests, client, async_client = mock_http
    chat = build_chat(kind, config=Settings(), chat_factory=partial(
        ChatOpenAI, http_client=client, http_async_client=async_client
    ))
    assert chat.request_timeout == 60
    assert chat.max_retries == 1
    assert chat.use_responses_api is False
    content = "请根据依据回答。" if kind == "qa" else [
        {"type": "text", "text": "读取图中的文字"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
    ]
    messages = [{"role": "user", "content": content}]
    assert chat.invoke(messages).content == "测试回复"
    request, body = requests[0]
    assert request.url.path == path
    assert body["messages"] == messages
    assert body["model"] == model_name
    assert body["temperature"] == temperature
    assert body["max_tokens"] == max_tokens
    assert body["thinking"] == {"type": "disabled"}
    assert "max_completion_tokens" not in body
    assert "extra_body" not in body
    assert len(requests) == 1


def test_embedding_wire_preserves_raw_strings_and_batches(monkeypatch, mock_http):
    from langchain_openai import OpenAIEmbeddings

    monkeypatch.setenv("MVP_EMBEDDING_API_KEY", "test-key")
    requests, client, async_client = mock_http
    embeddings = build_embeddings(config=Settings(), embeddings_factory=partial(
        OpenAIEmbeddings, http_client=client, http_async_client=async_client
    ))
    texts = [f"  中文原文 {i}\n<|endoftext|>  " for i in range(17)]
    vectors = embeddings.embed_documents(texts)
    assert len(vectors) == 17
    assert all(len(vector) == 1024 for vector in vectors)
    assert [len(body["input"]) for _, body in requests] == [16, 1]
    assert [text for _, body in requests for text in body["input"]] == texts
    assert all(body["model"] == "embedding-3" and body["dimensions"] == 1024 for _, body in requests)
    assert all(body["encoding_format"] == "float" for _, body in requests)
    assert all(request.url.path == "/api/paas/v4/embeddings" for request, _ in requests)
    assert embeddings.check_embedding_ctx_length is False
    assert len(embeddings.embed_query("  保留查询\n")) == 1024
    assert requests[-1][1]["input"] == ["  保留查询\n"]


@pytest.mark.parametrize("kind", ["qa", "embedding"])
def test_provider_failure_propagates_without_fake_result(monkeypatch, kind):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from openai import APIStatusError

    monkeypatch.setenv("MVP_QA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("MVP_EMBEDDING_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"message": "rejected", "type": "invalid_request_error"}}
    ))
    with httpx.Client(transport=transport) as client:
        async_client = httpx.AsyncClient(transport=transport)
        try:
            clients = {"http_client": client, "http_async_client": async_client}
            with pytest.raises(APIStatusError):
                if kind == "qa":
                    build_chat(config=Settings(), chat_factory=partial(ChatOpenAI, **clients)).invoke("question")
                else:
                    build_embeddings(config=Settings(), embeddings_factory=partial(OpenAIEmbeddings, **clients)).embed_query("question")
        finally:
            asyncio.run(async_client.aclose())


def test_data_paths_are_repo_relative_and_create_nothing(monkeypatch, tmp_path):
    from backend.src.config.data_paths import data_dir, database_path, uploads_dir

    monkeypatch.chdir(tmp_path)
    assert data_dir() == REPO_ROOT / "data" / "md-rag"
    monkeypatch.setenv("MVP_DATA_DIR", "custom-data")
    settings.clear_cache()
    assert data_dir() == REPO_ROOT / "custom-data"
    target = tmp_path / "document-data"
    monkeypatch.setenv("MVP_DATA_DIR", str(target))
    settings.clear_cache()
    assert database_path() == target / "documents.sqlite3"
    assert uploads_dir() == target / "uploads"
    assert not target.exists()


def test_existing_upload_and_sqlite_use_configured_data_root(monkeypatch, tmp_path):
    from backend.src.apps.services import state_store
    from backend.src.apps.services.common_service import ensure_upload_dir

    target = tmp_path / "documents"
    monkeypatch.setenv("MVP_DATA_DIR", str(target))
    assert ensure_upload_dir() == target / "uploads"
    record = {"doc_id": "test-document", "name": "sample.md", "status": "uploaded"}
    state_store.upsert_document(record)
    assert state_store.get_document("test-document") == record
    assert (target / "documents.sqlite3").is_file()
    assert not (target / "vector_index.sqlite3").exists()


def test_token_counter_counts_whitespace_and_special_text():
    from backend.src.chunking.token_counter import SimpleTokenCounter, count_tokens

    assert count_tokens("") == 0
    assert count_tokens("hello") == 1
    assert count_tokens("   \n") > 0
    assert count_tokens("<|endoftext|>") > 1
    assert SimpleTokenCounter().count("中英文 mixed") == count_tokens("中英文 mixed")


def test_tokenizer_failure_is_not_zero(monkeypatch):
    from backend.src.chunking import token_counter

    def failed_encode(*args, **kwargs):
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(token_counter, "_get_encoder", lambda: SimpleNamespace(encode=failed_encode))
    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        token_counter.count_tokens("非空原文")


def test_legacy_truncation_keeps_valid_unicode_prefix():
    from backend.src.chunking.token_counter import SimpleTokenCounter, count_tokens

    text = "这是中文字符和 emoji 🐱，mixed English。" * 3
    for budget in range(9):
        prefix = SimpleTokenCounter().truncate(text, budget)
        assert text.startswith(prefix)
        assert "�" not in prefix
        assert count_tokens(prefix) <= budget


@pytest.mark.parametrize("configured", [False, True])
def test_fresh_import_and_health_need_no_external_services(tmp_path, configured):
    # 新进程避免已缓存模块掩盖导入副作用；解释器就是启动 pytest 的 agent。
    script = '''
import importlib
import json
import socket
import sys
import tiktoken
import elasticsearch
import langchain_openai
from fastapi.testclient import TestClient

def forbidden(*args, **kwargs):
    raise AssertionError("import/health must not construct models or access external services")

# 保留 asyncio 在 Windows 建立内部 socketpair 的能力，拦截外部同步连接。
socket.create_connection = forbidden
tiktoken.get_encoding = forbidden
elasticsearch.Elasticsearch = forbidden
langchain_openai.ChatOpenAI = forbidden
langchain_openai.OpenAIEmbeddings = forbidden
from backend.src.main import app
importlib.import_module("backend.src.infrastructure.models")
assert "backend.src.apps.restful_apis.qa" not in sys.modules
with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.post("/auth/login", json={"username": "demo", "password": "demo123"}).status_code == 200
print(json.dumps(response.json()))
'''
    child_env = dict(os.environ)
    child_env["MVP_DATA_DIR"] = str(tmp_path / "unused-data")
    child_env["DEMO_USERNAME"] = "demo"
    child_env["DEMO_PASSWORD"] = "demo123"
    if configured:
        child_env["MVP_QA_LLM_API_KEY"] = "test-key"
        child_env["MVP_EMBEDDING_API_KEY"] = "test-key"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=child_env,
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "ok"}
    assert not (tmp_path / "unused-data").exists()
