from datetime import datetime, timedelta, timezone
import os
import time
from pathlib import Path

from fastapi import HTTPException,Request
import jwt

from backend.src.config.data_paths import uploads_dir

_TOKEN_ALG = "HS256"
_TOKEN_TTL_SECONDS = 24*60*60
_DEMO_ROLE = "authenticated"


"""
=============================
登录相关的
=============================
"""

def _demo_username()->str:
    return os.getenv("DEMO_USERNAME","demo")

def _demo_password()->str:
    return os.getenv("DEMO_PASSWORD","demo123")

def _jwt_secret()->str:
    return os.getenv("JWT_SECRET","jwt-secret")

def create_access_token(subject:str,role:str=_DEMO_ROLE)->str:
    exp = datetime.now(timezone.utc) + timedelta(seconds=_TOKEN_TTL_SECONDS)
    payload = {
        "sub" :subject,
        "role" : role,
        "exp":int(exp.timestamp()),
    }
    return jwt.encode(payload,_jwt_secret(),algorithm=_TOKEN_ALG)

def decode_access_token(token:str) ->dict:
    return jwt.decode(token,_jwt_secret(),algorithms=[_TOKEN_ALG])

def login_with_password(username:str, password:str)->dict:
    if username!= _demo_username() or password!= _demo_password():
        raise ServiceError("UNAUTHORIZED","用户名或密码错误",{"reason":"bad_credentials"})
    return{
        "access_token": create_access_token(subject=username),
        "token_type": "Bearer",
        "expires_in": _TOKEN_TTL_SECONDS,
        "role":_DEMO_ROLE,
    }

def _unauthorized(reason: str, request: Request | None = None) -> HTTPException:
    details = {"reason": reason}
    if request is not None:
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            details["request_id"] = request_id
    return HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "未授权访问", "details": details})


def require_authenticated(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise _unauthorized("missing_bearer", request)

    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise _unauthorized("empty_token", request)

    try:
        claims = decode_access_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("token_expired", request) from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("token_invalid", request) from exc

    if claims.get("role") != _DEMO_ROLE:
        raise _unauthorized("role_forbidden", request)

    return claims

"""
=============================
处理结果相关的
=============================
"""

class ServiceError(Exception):
    """
    code: 错误的标记
    message: 对于code的描述
    details：其他信息，比如request_id等
    """
    def __init__(self, code:str,message:str,details:dict|None = None)->None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def now_ms()->int:
    return int(time.time() * 1000)


def fail(code:str, message:str, details:dict|None = None) ->dict:
    payload = {
        "code" : code,
        "message" : message,
        "data": None,
    }

    if details is not None:
        payload["details"] = details
    return payload

def ok(data: dict|list|str|None=None, message: str="success")->dict:
    return{
        "code":"OK",
        "message":message,
        "data":data,
    }

def to_dict(record: dict) -> dict:
    """将内部记录（如文档元数据）转换为可供 API 直接返回的普通字典。

    返回浅拷贝，避免调用方修改返回值时污染内部存储。
    """
    return dict(record)



"""
=============================
文件上传相关的
=============================
"""

def get_upload_dir()->Path:
    return uploads_dir()

def ensure_upload_dir() ->Path:
    directory = get_upload_dir()
    directory.mkdir(parents=True,exist_ok=True)
    return directory


# =============================================================
# 演示：直接运行本文件（python common_service.py）时执行
# 展示 create_access_token 的返回值长什么样
# =============================================================
if __name__ == "__main__":
    token = create_access_token(subject="demo")

    print("=" * 60)
    print("1) create_access_token(...) 的返回值就是一段字符串：")
    print(token)
    print("=" * 60)

    header_b64, payload_b64, signature_b64 = token.split(".")
    print("2) 它由 3 段用 '.' 拼接而成（都是 base64url 编码，明文可读）：")
    print("   header    :", header_b64)
    print("   payload   :", payload_b64)
    print("   signature :", signature_b64)
    print("=" * 60)

    claims = decode_access_token(token)
    print("3) 服务端用同一把密钥 decode_access_token 解回来的 claims：")
    for k, v in claims.items():
        print(f"    {k:6}: {v}")
    print("=" * 60)
