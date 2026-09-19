from fastapi import APIRouter, Request, Response

from backend.src.apps.services.asset_service import AssetService
from backend.src.apps.services.common_service import require_authenticated


router = APIRouter(tags=["assets"])
service = AssetService()


@router.get("/assets/preview")
async def preview_asset(request: Request, document_id: str, asset_id: str) -> Response:
    """原图预览：凭 document_id/asset_id 二次鉴权后返回原文件字节。

    客户端不能提交服务器本地路径；storage_key 只在后端内部使用。
    """
    claims = require_authenticated(request)
    data = service.preview(
        document_id=document_id,
        asset_id=asset_id,
        owner_id=str(claims.get("sub") or "") or None,
    )
    return Response(content=data["content"], media_type=data["mime_type"])
