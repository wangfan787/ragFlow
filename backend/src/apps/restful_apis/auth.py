from datetime import datetime, timedelta, timezone
import os

from fastapi import APIRouter
import jwt
from pydantic import BaseModel

from backend.src.apps.services.common_service import ServiceError, login_with_password, ok

router = APIRouter(tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str



@router.post("/auth/login")
def login(req: LoginRequest) -> dict:
    data = login_with_password(req.username,req.password)
    return ok(data)
