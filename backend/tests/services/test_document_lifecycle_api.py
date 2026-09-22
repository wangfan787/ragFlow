"""文档生命周期 HTTP 端到端（替身入库管线，不依赖 ES/模型）。

覆盖 P0 闭环的 API 契约：
- 上传：UUID doc_id、SHA256/size/created_at 登记、file_path 不外泄、
  类型 422、超限 413；
- ingest：成功 ready+counts、失败 failed+502+error、并发 409、不存在 404；
- 详情/列表/源文件/删除/重试及对应状态码。
"""

from __future__ import annotations

import hashlib
import re

import pytest
from fastapi.testclient import TestClient

from backend.src.apps.restful_apis import documents as documents_api
from backend.src.config.settings import settings


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    # 上传与 SQLite 都落在临时目录；退出时清缓存避免污染其他测试
    monkeypatch.setitem(settings._data, "data_dir", str(tmp_path / "data"))
    yield tmp_path / "data"


@pytest.fixture
def client():
    from backend.src.apps.services.common_service import create_access_token
    from backend.src.main import create_app

    token = create_access_token("alice")
    return TestClient(create_app()), {"Authorization": f"Bearer {token}"}


class FakeIndexer:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)


class FakePipeline:
    """替身入库管线：记录调用，按 error 注入失败。"""

    def __init__(self):
        self.embedding_indexer = FakeIndexer()
        self.result = {"parsed_count": 2, "chunk_count": 5, "indexed_count": 5}
        self.error: str | None = None
        self.run_calls: list[str] = []

    def run(self, doc_id, parse_config, chunk_config, *, owner_id=None):
        self.run_calls.append(doc_id)
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.result)


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(documents_api.service, "_pipeline", fake)
    return fake


def _upload(client, headers, name="笔记.md", content=None):
    content = content if content is not None else "# 标题\n正文内容".encode("utf-8")
    return client.post(
        "/documents",
        files={"file": (name, content, "text/markdown")},
        headers=headers,
    )


def test_upload_uses_uuid_and_records_metadata(client):
    http, headers = client
    content = "# 标题\n正文内容".encode("utf-8")

    first = _upload(http, headers, content=content)
    assert first.status_code == 200
    data = first.json()["data"]
    assert re.fullmatch(r"[0-9a-f]{32}", data["doc_id"])
    assert data["sha256"] == hashlib.sha256(content).hexdigest()
    assert data["size"] == len(content)
    assert data["status"] == "uploaded"
    assert data["created_at"]
    assert "file_path" not in data

    second = _upload(http, headers, content=content)
    assert second.json()["data"]["doc_id"] != data["doc_id"]


def test_upload_rejects_unsupported_type(client):
    http, headers = client
    response = http.post(
        "/documents",
        files={"file": ("程序.exe", b"binary", "application/octet-stream")},
        headers=headers,
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_FILE_TYPE"


def test_upload_rejects_oversize(monkeypatch, client):
    http, headers = client
    monkeypatch.setitem(settings._data["upload"], "max_bytes", 10)
    response = _upload(http, headers, content=b"0123456789abc")
    assert response.status_code == 413
    body = response.json()
    assert body["code"] == "FILE_TOO_LARGE"
    assert body["details"]["max_bytes"] == 10


def test_detail_and_list_hide_internal_path(client):
    http, headers = client
    doc_id = _upload(http, headers).json()["data"]["doc_id"]

    detail = http.get(f"/documents/{doc_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["data"]["status"] == "uploaded"
    assert "file_path" not in detail.json()["data"]

    listing = http.get("/documents", headers=headers)
    ids = [item["doc_id"] for item in listing.json()["data"]["documents"]]
    assert ids == [doc_id]

    missing = http.get("/documents/does-not-exist", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["code"] == "DOC_NOT_FOUND"


def test_ingest_success_sets_ready_and_counts(client, fake_pipeline):
    http, headers = client
    doc_id = _upload(http, headers).json()["data"]["doc_id"]

    response = http.post("/documents/ingest", json={"doc_id": doc_id}, headers=headers)
    assert response.status_code == 200
    detail = http.get(f"/documents/{doc_id}", headers=headers).json()["data"]
    assert detail["status"] == "ready"
    assert detail["counts"] == {"parsed": 2, "chunk": 5, "indexed": 5}
    assert detail["error"] is None


def test_ingest_missing_doc_returns_404(client, fake_pipeline):
    http, headers = client
    response = http.post("/documents/ingest", json={"doc_id": "nope"}, headers=headers)
    assert response.status_code == 404


def test_ingest_failure_marks_failed_then_retry_recovers(client, fake_pipeline):
    http, headers = client
    doc_id = _upload(http, headers).json()["data"]["doc_id"]
    fake_pipeline.error = "embedding 服务不可用"

    failed = http.post("/documents/ingest", json={"doc_id": doc_id}, headers=headers)
    assert failed.status_code == 502
    assert failed.json()["code"] == "INGEST_FAILED"
    detail = http.get(f"/documents/{doc_id}", headers=headers).json()["data"]
    assert detail["status"] == "failed"
    assert "embedding 服务不可用" in detail["error"]

    fake_pipeline.error = None
    retry = http.post(f"/documents/{doc_id}/retry", headers=headers)
    assert retry.status_code == 200
    detail = http.get(f"/documents/{doc_id}", headers=headers).json()["data"]
    assert detail["status"] == "ready"


def test_concurrent_ingest_returns_409(client, fake_pipeline):
    http, headers = client
    doc_id = _upload(http, headers).json()["data"]["doc_id"]
    with documents_api.service._indexing_guard:
        documents_api.service._indexing.add(doc_id)
    try:
        response = http.post("/documents/ingest", json={"doc_id": doc_id}, headers=headers)
        assert response.status_code == 409
        assert response.json()["code"] == "DOC_INDEXING_IN_PROGRESS"
    finally:
        with documents_api.service._indexing_guard:
            documents_api.service._indexing.discard(doc_id)


def test_source_download_roundtrip(client):
    http, headers = client
    content = "# 标题\n正文内容".encode("utf-8")
    doc_id = _upload(http, headers, content=content).json()["data"]["doc_id"]

    response = http.get(f"/documents/{doc_id}/source", headers=headers)
    assert response.status_code == 200
    assert response.content == content

    missing = http.get("/documents/does-not-exist/source", headers=headers)
    assert missing.status_code == 404


def test_delete_cleans_record_file_and_index(client, fake_pipeline, isolated_data_dir):
    http, headers = client
    doc_id = _upload(http, headers).json()["data"]["doc_id"]

    response = http.delete(f"/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"] == {"doc_id": doc_id, "deleted": True}

    assert fake_pipeline.embedding_indexer.deleted == [doc_id]
    assert http.get(f"/documents/{doc_id}", headers=headers).status_code == 404
    uploads = (isolated_data_dir / "uploads").glob(f"{doc_id}.*")
    assert list(uploads) == []

    again = http.delete(f"/documents/{doc_id}", headers=headers)
    assert again.status_code == 404
