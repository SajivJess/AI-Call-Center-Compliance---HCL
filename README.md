# AI Call Center Compliance API

Production-ready FastAPI service for the HCL Call Center Compliance challenge.

## Problem Focus

This API accepts Base64 audio and returns strict JSON analytics with:

- Transcript
- Summary
- SOP compliance checks and score
- Payment and rejection classification
- Sentiment
- Keywords

## Core Features

- Endpoint: `POST /api/call-analytics`
- API-key auth via `x-api-key`
- Global rate limiting via `slowapi`
- Always-on audio normalization: FFmpeg -> mono 16k WAV + `highpass`, `lowpass`, `dynaudnorm`
- STT fallback: Sarvam (primary) -> Whisper (fallback)
- LLM fallback: OpenRouter (primary) -> Gemini (fallback)
- Strict contract validation with Pydantic
- Language output normalized to `Tamil` or `Hindi`
- Summary placeholder protection (no mock/placeholder text in output)
- Domain-aware keyword extraction for Tamil/Hinglish call-center context

## Architecture Overview

1. Request auth + validation
2. Base64 decode
3. FFmpeg preprocessing
4. STT transcription
5. LLM normalization + summary
6. SOP validation + analytics + keyword extraction
7. Contract validation + JSON response
8. SQLite persistence + optional FAISS indexing

## Endpoint Tester: How To Use

Use the endpoint tester to validate your deployment behavior before submission.

1. Enter deployed API URL
2. Provide authorization/API key in headers
3. Click `Test Endpoint`

What this tests:

- Authentication headers
- Audio input processing
- Request parsing/validation
- JSON response formatting
- API stability/behavior

Note: the official evaluation uses a separate automated system and official audio samples.

## API Contract

### Request

```json
{
  "audio_base64": "<base64-mp3>",
  "audio_format": "mp3",
  "language_hint": "hi-en",
  "call_id": "CALL-123"
}
```

### Response (shape)

- `status`
- `language`
- `callId`
- `transcript`
- `summary`
- `sop_validation`
- `analytics`
- `sentiment`
- `keywords`
- `modelInfo`

## Local Setup

1. Create and activate virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Copy env template and configure keys

```bash
cp .env.example .env
```

4. Run server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

5. Health check

```bash
curl http://127.0.0.1:8000/health
```

## Deployment

This repo includes:

- `Procfile` for simple PaaS startup
- `runtime.txt` for Python runtime pinning

Recommended free-tier options:

- Render
- Railway
- Fly.io
- Vercel (API hosting pattern)
- AWS Free Tier

Startup command used by platform:

```bash
uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

Deployment checklist:

1. Set env vars from `.env.example`
2. Ensure FFmpeg is available in runtime
3. Keep live URL accessible for at least 48 hours after deadline

## Submission Requirements

- Live deployed public URL
- Public GitHub repository
- Public demo video link (YouTube or Google Drive)
- AI tools disclosure in README
- Optional slide deck

Video demo requirements:

- 2 to 5 minutes
- Screen recording with narration
- Show key features end-to-end

## AI Tools Used

- GitHub Copilot (GPT-5.3-Codex) for implementation support, refactoring, and test scaffolding

## Known Limitations

- External provider latency (STT/LLM) affects response time
- Accuracy depends on source audio quality and language mixing
- SQLite is used for lightweight persistence (not horizontal-scale primary DB)

## Tests

Run targeted checks:

```bash
pytest tests/test_api_contract.py tests/test_audio_and_language.py -q
```
