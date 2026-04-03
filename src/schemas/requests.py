from pydantic import BaseModel, Field


class CallAnalyticsRequest(BaseModel):
    audio_base64: str = Field(..., min_length=32)
    audio_format: str = Field(default="mp3", pattern="^(mp3|wav|m4a|ogg)$")
    language_hint: str | None = Field(default=None, pattern="^(hi|en|ta|hi-en|ta-en)$")
    call_id: str | None = Field(default=None, max_length=64)
