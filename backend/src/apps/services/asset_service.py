"""图片资产预览：读取前用同一条归属查询再次鉴权（P0-A）。

预览与解析共用同一条授权路径：asset_id + document_id + 当前用户
必须同时命中 assets/documents JOIN；storage_key 只信数据库里保存的
值，且读取前确认规范化路径仍在资产根目录内。
"""

from __future__ import annotations

from fastapi import HTTPException

from backend.src.assets.asset_store import (
    AssetFileStore,
    AssetNotFoundError,
    AssetPathError,
    AssetRegistry,
)


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": {}},
    )


class AssetService:
    def __init__(
        self, registry: AssetRegistry | None = None, file_store: AssetFileStore | None = None,
    ) -> None:
        self.registry = registry or AssetRegistry()
        self.file_store = file_store or AssetFileStore()

    def preview(self, *, document_id: str, asset_id: str, owner_id: str | None) -> dict:
        if not document_id or not asset_id:
            raise _error(400, "INVALID_ASSET_REQUEST", "document_id 与 asset_id 不能为空")
        row = self.registry.authorized_asset(asset_id, document_id, owner_id)
        if row is None:
            # 区分"资源不存在"与"无权限"便于排查；两者都不返回文件内容
            if self.registry.get_asset(asset_id, document_id) is None:
                raise _error(404, "ASSET_NOT_FOUND", "资源不存在或不属于该文档")
            raise _error(403, "ASSET_FORBIDDEN", "当前用户无权访问该资源")
        try:
            content = self.file_store.read(row["storage_key"])
        except AssetPathError as exc:
            raise _error(400, "ASSET_PATH_INVALID", f"存储键非法: {exc}") from exc
        except AssetNotFoundError as exc:
            raise _error(404, "ASSET_FILE_MISSING", f"资产文件缺失: {exc}") from exc
        return {"content": content, "mime_type": row["mime_type"], "asset_id": asset_id}
