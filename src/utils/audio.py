import base64
import os
import subprocess
import uuid
from pathlib import Path

from fastapi import HTTPException

from src.config.settings import get_settings


def _ensure_temp_dir() -> Path:
    settings = get_settings()
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def decode_base64_audio(audio_base64: str, extension: str = "mp3") -> Path:
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid Base64 audio") from exc

    temp_dir = _ensure_temp_dir()
    source = temp_dir / f"{uuid.uuid4().hex}.{extension}"
    source.write_bytes(audio_bytes)
    return source


def preprocess_audio(input_path: Path) -> Path:
    settings = get_settings()
    output_path = input_path.with_suffix(".wav")
    command = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-af",
        "highpass=f=200,lowpass=f=3000,dynaudnorm",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise HTTPException(status_code=422, detail=f"Audio preprocessing failed: {stderr[:200]}") from exc
    return output_path


def cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path and path.exists():
                os.remove(path)
        except OSError:
            pass
