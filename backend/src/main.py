import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.src.apps.restful_apis.auth import router as auth_router
from backend.src.apps.restful_apis.documents import router as documents_router
from backend.src.apps.services.common_service import (
    ServiceError,
    fail,
    now_ms,
)
from backend.src.infrastructure.models import ModelConfigurationError


def create_app() -> FastAPI:
    app = FastAPI(title="rag-mvp-mixed-strategy-api", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_trace_middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.start_ms = now_ms()
        response = await call_next(request)
        latency_ms = max(0, now_ms() - request.state.start_ms)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Stage"] = "mvp"
        response.headers["X-Latency-Ms"] = str(latency_ms)
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        details = dict(exc.details)
        details.setdefault("request_id", request.state.request_id)
        status_code = 401 if exc.code == "UNAUTHORIZED" else 400
        return JSONResponse(status_code=status_code, content=fail(exc.code, exc.message, details))

    @app.exception_handler(ModelConfigurationError)
    async def model_configuration_error_handler(request: Request, exc: ModelConfigurationError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and {"code", "message", "details"}.issubset(exc.detail):
            details = dict(exc.detail.get("details", {}))
            details.setdefault("request_id", request.state.request_id)
            return JSONResponse(
                status_code=exc.status_code,
                content=fail(str(exc.detail["code"]), str(exc.detail["message"]), details),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=fail("HTTP_ERROR", str(exc.detail), {"request_id": request.state.request_id}),
        )

    @app.exception_handler(Exception)
    async def fallback_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=fail("INTERNAL_ERROR", str(exc), {"request_id": request.state.request_id}),
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(auth_router, prefix="")
    app.include_router(documents_router, prefix="")
    # QA 路由待阶段 07 按最终 API 挂载。
    return app


app = create_app()
