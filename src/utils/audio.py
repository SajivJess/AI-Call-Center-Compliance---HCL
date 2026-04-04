import base64
import os
import subprocess
import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException

from src.config.settings import get_settings


def _ensure_temp_dir() -> Path:
    settings = get_settings()
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _resolve_ffmpeg_path() -> str:
    settings = get_settings()
    candidates: list[str] = []

    if settings.ffmpeg_binary:
        candidates.append(settings.ffmpeg_binary)
    if settings.ffmpeg_binary != "ffmpeg":
        candidates.append("ffmpeg")
    candidates.extend(["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"])

    for candidate in candidates:
        if os.path.isabs(candidate):
            if Path(candidate).exists():
                return candidate
        else:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

    return settings.ffmpeg_binary or "ffmpeg"


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
    ffmpeg_path = _resolve_ffmpeg_path()
    print("FFmpeg path:", ffmpeg_path)
    print("Audio input exists:", input_path.exists())
    if input_path.exists():
        print("Audio input size:", input_path.stat().st_size)
    command = [
        ffmpeg_path,
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
    print("Audio output exists:", output_path.exists())
    if output_path.exists():
        print("Audio output size:", output_path.stat().st_size)
    return output_path


def split_audio_into_chunks(input_path: Path, max_seconds: int = 29) -> list[Path]:
    ffmpeg_path = _resolve_ffmpeg_path()

    chunk_dir = _ensure_temp_dir() / f"{input_path.stem}_chunks_{uuid.uuid4().hex}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunk_dir / "chunk_%03d.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-f",
        "segment",
        "-segment_time",
        str(max_seconds),
        "-reset_timestamps",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_pattern),
    ]
    print("Splitting audio into chunks:", str(chunk_dir))
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise HTTPException(status_code=422, detail=f"Audio chunking failed: {stderr[:200]}") from exc

    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    print("Chunk count:", len(chunks))
    for chunk in chunks:
        print("Chunk size:", chunk.name, chunk.stat().st_size if chunk.exists() else 0)
    if not chunks:
        raise HTTPException(status_code=422, detail="Audio chunking failed: no chunks produced")
    return chunks


def cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path and path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        except OSError:
            pass
