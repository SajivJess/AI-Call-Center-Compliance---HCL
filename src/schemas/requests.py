from pydantic import BaseModel, ConfigDict, Field


class CallAnalyticsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    audio_base64: str = Field(..., min_length=32, alias="audioBase64")
    audio_format: str = Field(default="mp3", pattern="^(mp3|wav|m4a|ogg)$", alias="audioFormat")
    language_hint: str | None = Field(default=None, pattern="^(hi|en|ta|hi-en|ta-en)$", alias="languageHint")
    call_id: str | None = Field(default=None, max_length=64, alias="callId")
