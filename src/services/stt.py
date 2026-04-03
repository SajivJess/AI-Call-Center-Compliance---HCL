from pathlib import Path

from fastapi import HTTPException
from openai import OpenAI
from sarvamai import SarvamAI

from src.config.settings import get_settings


class STTService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _normalize_language_hint(self, language_hint: str | None) -> str | None:
        if language_hint in {"hi-en", "ta-en"}:
            return "hi" if language_hint.startswith("hi") else "ta"
        if language_hint in {"hi", "ta", "en"}:
            return language_hint
        return None

    def _looks_mojibake(self, text: str) -> bool:
        markers = ("à®", "à¯", "Ã", "â", "\ufffd")
        return any(marker in text for marker in markers)

    def _normalize_transcript_text(self, text: str) -> str:
        cleaned = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        if self._looks_mojibake(cleaned):
            try:
                repaired = cleaned.encode("latin1").decode("utf-8")
                if repaired.strip():
                    cleaned = repaired
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        return cleaned.strip()

    def transcribe(self, wav_path: Path, language_hint: str | None = None) -> tuple[str, str]:
        errors: list[str] = []
        print("Sarvam key:", bool(self.settings.sarvam_api_key))
        print("Whisper key:", bool(self.settings.whisper_api_key))

        if self.settings.sarvam_api_key:
            try:
                return self._sarvam_transcribe(wav_path, language_hint), "sarvam"
            except Exception as exc:
                errors.append(f"sarvam:{exc}")

        if self.settings.whisper_api_key:
            try:
                return self._whisper_transcribe(wav_path, language_hint), "whisper"
            except Exception as exc:
                errors.append(f"whisper:{exc}")

        if self.settings.allow_mock_stt:
            return self._mock_transcribe(), "mock"

        raise HTTPException(status_code=502, detail=f"No STT provider available. {' | '.join(errors)}")

    def _sarvam_transcribe(self, wav_path: Path, language_hint: str | None) -> str:
        print("Sending audio to Sarvam...")
        client = SarvamAI(api_subscription_key=self.settings.sarvam_api_key)
        with wav_path.open("rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe",
            )
        print("Sarvam response:", response)
        text = getattr(response, "transcript", None) or (response.get("transcript") if isinstance(response, dict) else None)
        if not text:
            raise ValueError("STT returned empty — investigate Sarvam response")
        return self._normalize_transcript_text(text)

    def _whisper_transcribe(self, wav_path: Path, language_hint: str | None) -> str:
        client = OpenAI(api_key=self.settings.whisper_api_key)
        print("Sending audio to Whisper...")
        with wav_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=self.settings.whisper_model,
                file=audio_file,
                language=self._normalize_language_hint(language_hint),
            )
        print("Whisper response:", getattr(result, "text", ""))
        text = getattr(result, "text", "")
        if not text:
            raise ValueError("STT returned empty — investigate Whisper response")
        return self._normalize_transcript_text(text)

    def _mock_transcribe(self) -> str:
        return "Hello sir, this is a follow-up call. I will explain the plan and next payment options."
