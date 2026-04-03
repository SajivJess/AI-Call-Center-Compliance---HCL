import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.schemas.requests import CallAnalyticsRequest
from src.schemas.responses import CallAnalyticsResponse
from src.services.pipeline import CallAnalyticsPipeline
from src.storage.sqlite_store import SQLiteStore
from src.utils.auth import validate_api_key

router = APIRouter(prefix="/api", tags=["Call Analytics"])

pipeline = CallAnalyticsPipeline()
store = SQLiteStore()


@router.post("/call-analytics", response_model=CallAnalyticsResponse)
async def call_analytics(payload: CallAnalyticsRequest, _: str = Depends(validate_api_key)) -> CallAnalyticsResponse:
    result = await pipeline.run_async(payload)
    await asyncio.to_thread(store.upsert_result, result)
    return JSONResponse(content=result.model_dump(mode="json"), media_type="application/json; charset=utf-8")
