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

from backend.src.config.settings import Settings, settings, DEFAULT_CONFIG, _merge
from backend.src.infrastructure.models import (
    ModelConfigurationError,
    build_chat,
    build_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _set_config(name, value):
    patch = {}
    section = patch
    parts = name.split(".")
    for part in parts[:-1]:
        section = section.setdefault(part, {})
    section[parts[-1]] = value
    settings._data = _merge(settings._data, patch)


def _settings():
    return Settings(local_path=None, overrides=settings._data)


def test_yaml_priority_and_environment_is_ignored(monkeypatch, tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text('embedding:\n  model: local-model # 可注释\n', encoding="utf-8")
    monkeypatch.setenv("MVP_EMBEDDING_MODEL", "shell-model")
    monkeypatch.setenv("MVP_CONFIG_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("MVP_ENV_FILE", str(tmp_path / ".env"))
    (tmp_path / ".env").write_text("MVP_EMBEDDING_MODEL=dotenv-model\n")
    assert Settings(local_path=local).text("embedding.model") == "local-model"
    assert Settings(local_path=None).text("embedding.model") == "embedding-3"
    assert Settings(local_path=local, overrides={"embedding": {"model": "explicit"}}).text("embedding.model") == "explicit"


def test_unknown_and_invalid_yaml_fields_fail_without_leaking_values(tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text('embedding:\n  api_keey: private-marker\n')
    with pytest.raises(ValueError, match="embedding.api_keey") as error:
        Settings(local_path=local)
    assert "private-marker" not in str(error.value)
    local.write_text('embedding: [private-marker')
    with pytest.raises(ValueError, match="Invalid YAML") as error:
        Settings(local_path=local)
    assert "private-marker" not in str(error.value)
    with pytest.raises(ValueError, match="integer"):
        Settings(local_path=None, overrides={"embedding": {"dimensions": "1024"}}).integer("embedding.dimensions")


def test_existing_model_and_es_settings():
    config = Settings(local_path=None, overrides={
        "embedding": {"model": "custom-embedding", "dimensions": 256, "batch_size": 3},
        "llm": {
            "qa": {"model": "custom-qa", "base_url": "https://qa.example/v1", "api_key": "qa-key"},
            "vision": {"model": "custom-vision", "base_url": "https://vision.example/v1", "api_key": "vision-key"},
        },
        "elasticsearch": {"url": "http://es.example:9200", "index": "existing-index", "timeout": 12},
    })
    assert config.text("embedding.model") == "custom-embedding"
    assert config.integer("embedding.dimensions") == 256
    qa = build_chat("qa", config=config, chat_factory=lambda **kw: kw)
    assert (qa["model"], qa["base_url"], qa["api_key"]) == ("custom-qa", "https://qa.example/v1", "qa-key")
    vision = build_chat("vision", config=config, chat_factory=lambda **kw: kw)
    assert vision["api_key"] == "vision-key"
    assert config.text("elasticsearch.index") == "existing-index"
    assert config.integer("elasticsearch.timeout") == 12


def test_missing_keys_do_not_use_environment_or_other_services(monkeypatch):
    monkeypatch.setenv("MVP_QA_LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    config = Settings(local_path=None, overrides={"embedding": {"api_key": "embedding-only"}})
    for kind in ("qa", "vision"):
        with pytest.raises(ModelConfigurationError, match=f"llm.{kind}.api_key"):
            build_chat(kind, config=config)
    with pytest.raises(ModelConfigurationError, match="embedding.api_key"):
        build_embeddings(config=Settings(local_path=None))


def test_invalid_model_configuration_fails_before_construction():
    config = Settings(local_path=None, overrides={"llm": {"qa": {"context_limit_tokens": 1280}}})
    with pytest.raises(ModelConfigurationError, match="总上下文"):
        build_chat("qa", config=config)
    with pytest.raises(ValueError, match="qa 或 vision"):
        build_chat("metadata")
    config = Settings(local_path=None, overrides={"embedding": {"api_key": "test", "batch_size": 17}})
    with pytest.raises(ValueError, match="embedding.batch_size"):
        build_embeddings(config=config)


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

    _set_config(f"llm.{kind}.api_key", "test-key")
    requests, client, async_client = mock_http
    chat = build_chat(kind, config=_settings(), chat_factory=partial(
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

    _set_config("embedding.api_key", "test-key")
    requests, client, async_client = mock_http
    embeddings = build_embeddings(config=_settings(), embeddings_factory=partial(
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

    _set_config("llm.qa.api_key", "test-key")
    _set_config("embedding.api_key", "test-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"message": "rejected", "type": "invalid_request_error"}}
    ))
    with httpx.Client(transport=transport) as client:
        async_client = httpx.AsyncClient(transport=transport)
        try:
            clients = {"http_client": client, "http_async_client": async_client}
            with pytest.raises(APIStatusError):
                if kind == "qa":
                    build_chat(config=_settings(), chat_factory=partial(ChatOpenAI, **clients)).invoke("question")
                else:
                    build_embeddings(config=_settings(), embeddings_factory=partial(OpenAIEmbeddings, **clients)).embed_query("question")
        finally:
            asyncio.run(async_client.aclose())


def test_data_paths_are_repo_relative_and_create_nothing(monkeypatch, tmp_path):
    from backend.src.config.data_paths import data_dir, database_path, uploads_dir

    monkeypatch.chdir(tmp_path)
    _set_config("data_dir", "data/md-rag")
    assert data_dir() == REPO_ROOT / "data" / "md-rag"
    _set_config("data_dir", "custom-data")
    assert data_dir() == REPO_ROOT / "custom-data"
    target = tmp_path / "document-data"
    _set_config("data_dir", str(target))
    assert database_path() == target / "documents.sqlite3"
    assert uploads_dir() == target / "uploads"
    assert not target.exists()


def test_existing_upload_and_sqlite_use_configured_data_root(monkeypatch, tmp_path):
    from backend.src.apps.services import state_store
    from backend.src.apps.services.common_service import ensure_upload_dir

    target = tmp_path / "documents"
    _set_config("data_dir", str(target))
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
from backend.src.config.settings import Settings, settings
settings._data = Settings(local_path=None, overrides=json.loads(sys.argv[1]))._data
from backend.src.main import app
importlib.import_module("backend.src.infrastructure.models")
# QA 路由已挂载：导入可以发生，但上面的 forbidden 钩子保证它不得构造模型或连接外部服务
assert "/qa/query" in app.openapi()["paths"]
with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.post("/auth/login", json={"username": "demo", "password": "demo123"}).status_code == 200
print(json.dumps(response.json()))
'''
    config = {"data_dir": str(tmp_path / "unused-data")}
    if configured:
        config.update({"embedding": {"api_key": "test-key"}, "llm": {"qa": {"api_key": "test-key"}}})
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(config)], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "ok"}
    assert not (tmp_path / "unused-data").exists()
