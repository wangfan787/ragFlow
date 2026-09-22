from datetime import datetime, timedelta, timezone
import time
from pathlib import Path

from fastapi import HTTPException,Request
import jwt

from backend.src.config.data_paths import uploads_dir
from backend.src.config.settings import settings

_TOKEN_ALG = "HS256"
_DEMO_ROLE = "authenticated"


"""
=============================
登录相关的
=============================
"""

def _demo_username()->str:
    return settings.text("auth.username")

def _demo_password()->str:
    return settings.text("auth.password")

def _jwt_secret()->str:
    return settings.text("auth.jwt_secret")

def create_access_token(subject:str,role:str=_DEMO_ROLE)->str:
    exp = datetime.now(timezone.utc) + timedelta(seconds=settings.integer("auth.token_ttl_seconds", positive=True))
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
        raise ServiceError("UNAUTHORIZED","用户名或密码错误",{"reason":"bad_credentials"},status_code=401)
    return{
        "access_token": create_access_token(subject=username),
        "token_type": "Bearer",
        "expires_in": settings.integer("auth.token_ttl_seconds", positive=True),
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
    status_code: 映射到 HTTP 状态码（400/404/409/413/422/502 等），
        由 main.py 的全局处理器直接使用；未显式指定时按请求错误 400 处理。
    """
    def __init__(self, code:str,message:str,details:dict|None = None,*,status_code:int = 400)->None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code


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
