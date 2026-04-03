import base64
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.main import app


def main() -> None:
    settings = get_settings()
    client = TestClient(app)

    chunks_dir = PROJECT_ROOT / "tests audio" / "chunks"
    output_dir = PROJECT_ROOT / "tests audio" / "outputs_rerun7"
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_name in ["chunk_000.mp3", "chunk_001.mp3"]:
        chunk_path = chunks_dir / chunk_name
        payload = {
            "audio_base64": base64.b64encode(chunk_path.read_bytes()).decode("ascii"),
            "audio_format": "mp3",
            "language_hint": "ta-en" if chunk_name == "chunk_001.mp3" else "hi-en",
            "call_id": f"RERUN7-{chunk_path.stem}",
        }
        response = client.post(
            "/api/call-analytics",
            headers={"x-api-key": settings.api_key},
            json=payload,
        )
        response.raise_for_status()
        output_path = output_dir / f"{chunk_path.stem}_response.json"
        output_path.write_text(json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
