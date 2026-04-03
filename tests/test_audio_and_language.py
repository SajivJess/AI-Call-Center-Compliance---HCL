from pathlib import Path

from src.services.pipeline import CallAnalyticsPipeline
from src.utils.audio import preprocess_audio


def test_preprocess_audio_applies_cleanup_filters(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"dummy-audio")

    captured = {}

    def _mock_run(command, check, capture_output):
        captured["command"] = command
        captured["check"] = check
        captured["capture_output"] = capture_output

    monkeypatch.setattr("src.utils.audio.subprocess.run", _mock_run)

    output_path = preprocess_audio(input_path)

    assert output_path.suffix == ".wav"
    assert captured["check"] is True
    assert captured["capture_output"] is True
    assert "-af" in captured["command"]
    assert "highpass=f=200,lowpass=f=3000,dynaudnorm" in captured["command"]


def test_canonical_language_detection() -> None:
    pipeline = CallAnalyticsPipeline()

    assert pipeline._canonical_language("வணக்கம், EMI amount iruku.") == "Tamil"
    assert pipeline._canonical_language("Hello sir, your payment is pending.") == "Hindi"
    assert pipeline._canonical_language("Hello sir, your payment is pending.", "ta-en") == "Tamil"