"""P0-A：资产预览二次鉴权（服务层矩阵 + HTTP 端点）。

- 有权限：owner 命中 JOIN → 200 返回原图字节；
- 无权限：其他用户 / 空 owner → 403；
- 失效资源：资产行不存在 → 404；文件被删 → 404；
- 路径穿越：被篡改的 storage_key → 400，绝不读资产根目录之外的文件。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.src.apps.services.asset_service import AssetService
from backend.src.assets.asset_store import AssetFileStore, AssetRegistry

PNG_BYTES = b"\x89PNG\r\n\x1a\npreview-bytes"


def _setup(tmp_path: Path, owner: str = "alice") -> tuple[AssetService, AssetFileStore, AssetRegistry, dict]:
    db_path = tmp_path / "db.sqlite3"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, payload) VALUES (?, ?)",
            ("doc_1", json.dumps({"doc_id": "doc_1", "owner_id": owner})),
        )
    store = AssetFileStore(root=tmp_path / "assets")
    registry = AssetRegistry(db_path=db_path)
    asset_id, storage_key = store.save(PNG_BYTES, ".png")
    registry.register(
        asset_id=asset_id,
        document_id="doc_1",
        storage_key=storage_key,
        mime_type="image/png",
        sha256=asset_id.removeprefix("sha256:"),
        size_bytes=len(PNG_BYTES),
        owner_id=owner,
    )
    return AssetService(registry=registry, file_store=store), store, registry, {
        "asset_id": asset_id, "storage_key": storage_key, "db_path": db_path,
    }


def test_owner_can_preview(tmp_path: Path) -> None:
    service, _, _, fixture = _setup(tmp_path)
    data = service.preview(document_id="doc_1", asset_id=fixture["asset_id"], owner_id="alice")
    assert data["content"] == PNG_BYTES
    assert data["mime_type"] == "image/png"


def test_other_user_is_forbidden(tmp_path: Path) -> None:
    service, _, _, fixture = _setup(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_1", asset_id=fixture["asset_id"], owner_id="mallory")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["code"] == "ASSET_FORBIDDEN"
    # 空 owner（离线/遗留文档）同样无权访问
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_1", asset_id=fixture["asset_id"], owner_id=None)
    assert exc_info.value.status_code == 403


def test_unknown_asset_returns_404(tmp_path: Path) -> None:
    service, _, _, _ = _setup(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_1", asset_id="sha256:missing", owner_id="alice")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == "ASSET_NOT_FOUND"
    # 文档不匹配也按 404 处理（资源不属于该文档）
    service, _, _, fixture = _setup(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_other", asset_id=fixture["asset_id"], owner_id="alice")
    assert exc_info.value.status_code == 404


def test_deleted_file_returns_404(tmp_path: Path) -> None:
    service, store, _, fixture = _setup(tmp_path)
    (store.root / fixture["storage_key"]).unlink()
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_1", asset_id=fixture["asset_id"], owner_id="alice")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == "ASSET_FILE_MISSING"


def test_tampered_storage_key_is_rejected(tmp_path: Path) -> None:
    service, _, _, fixture = _setup(tmp_path)
    with sqlite3.connect(str(fixture["db_path"])) as conn:
        conn.execute(
            "UPDATE assets SET storage_key = ? WHERE asset_id = ?",
            ("../../etc/passwd", fixture["asset_id"]),
        )
    with pytest.raises(HTTPException) as exc_info:
        service.preview(document_id="doc_1", asset_id=fixture["asset_id"], owner_id="alice")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "ASSET_PATH_INVALID"


def test_preview_endpoint_enforces_auth(tmp_path: Path, monkeypatch) -> None:
    from backend.src.apps.restful_apis import assets as assets_api
    from backend.src.apps.services.common_service import create_access_token
    from backend.src.main import create_app

    service, _, _, fixture = _setup(tmp_path)
    monkeypatch.setattr(assets_api.service, "registry", service.registry)
    monkeypatch.setattr(assets_api.service, "file_store", service.file_store)
    client = TestClient(create_app())
    params = {"document_id": "doc_1", "asset_id": fixture["asset_id"]}

    # 未带 token → 401
    response = client.get("/assets/preview", params=params)
    assert response.status_code == 401

    # owner → 200 + 原图字节
    token = create_access_token("alice")
    response = client.get(
        "/assets/preview", params=params, headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"].startswith("image/png")

    # 其他用户 → 403
    other = create_access_token("mallory")
    response = client.get(
        "/assets/preview", params=params, headers={"Authorization": f"Bearer {other}"},
    )
    assert response.status_code == 403
