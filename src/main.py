from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config.settings import get_settings
from src.routes.call_analytics import router as analytics_router

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])

app = FastAPI(title=settings.app_name)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}))
app.add_middleware(SlowAPIMiddleware)

BASE_DIR = Path(__file__).resolve().parent
DEMO_HTML = BASE_DIR / "static" / "demo.html"


@app.get("/")
def root_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.get("/demo")
def live_demo() -> FileResponse:
    return FileResponse(DEMO_HTML)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id", "")
    response = await call_next(request)
    if request.state.request_id:
        response.headers["x-request-id"] = request.state.request_id
    return response


app.include_router(analytics_router)
