from fastapi import Header, HTTPException

from src.config.settings import get_settings


async def validate_api_key(x_api_key: str = Header(default="", alias="x-api-key")) -> str:
    settings = get_settings()
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
