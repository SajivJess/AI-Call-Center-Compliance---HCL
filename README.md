# AI Call Center Compliance API

FastAPI service for the HCL Track 3 call center compliance challenge.

## Description

This API accepts one MP3 call recording at a time via Base64, runs multi-stage AI analysis, and returns structured JSON for:

- Transcript
- Summary
- SOP compliance checks and score
- Payment preference classification
- Rejection reason detection
- Sentiment
- Keywords

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic v2
- Uvicorn
- Sarvam AI SDK for speech-to-text
- OpenRouter + Gemini for LLM fallback
- FFmpeg for audio preprocessing
- SQLite for persistence
- FAISS for vector indexing
- SlowAPI for rate limiting

## Approach

1. Validate request and API key.
2. Decode Base64 MP3 input.
3. Normalize audio with FFmpeg.
4. Split long audio into 29-second chunks.
5. Transcribe each chunk and merge the transcripts.
6. Run LLM normalization and summarization.
7. Extract SOP, analytics, sentiment, and keywords.
8. Validate the final response contract and return JSON.

## Architecture Overview

1. `POST /api/call-analytics` receives Base64 MP3 input.
2. `src/utils/audio.py` decodes, preprocesses, and chunks audio.
3. `src/services/stt.py` transcribes chunks with Sarvam.
4. `src/services/llm.py` normalizes the merged transcript and builds the summary.
5. `src/services/pipeline.py` merges all outputs, validates the response, and indexes the transcript in FAISS.
6. `src/services/sop.py`, `src/services/analytics.py`, and `src/services/nlp.py` produce compliance and business-intelligence fields.

## Core Features

- Endpoint: `POST /api/call-analytics`
- Mandatory API key auth via `x-api-key`
- Audio normalization: FFmpeg -> mono 16k WAV + `highpass`, `lowpass`, `dynaudnorm`
- Chunked transcription for long calls
- Transcript merge before NLP/LLM analysis
- Strict response validation with Pydantic
- Tamil/Hindi language normalization
- No mock/placeholder summaries in production responses
- Domain-aware keyword extraction
- Semantic indexing of transcripts with FAISS

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

### Response Fields

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

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy the environment template.

```bash
cp .env.example .env
```

4. Run the server.

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

5. Verify health.

```bash
curl http://127.0.0.1:8000/health
```

## Deployment

This repo supports Railway, Render, Fly.io, and similar PaaS platforms.

Included deployment files:

- `Dockerfile`
- `Procfile`
- `runtime.txt`

Deployment checklist:

1. Set environment variables from `.env.example` in the host platform.
2. Ensure FFmpeg is available in the runtime.
3. Set `ALLOW_MOCK_STT=false` and `ALLOW_MOCK_LLM=false`.
4. For full-chunk judging, set `MAX_SYNC_CHUNKS` to at least the expected chunk count (example: `20`). Set `MAX_SYNC_CHUNKS=0` to disable capping.
5. Keep the live URL public and available for 48 hours after submission.

## Submission Checklist

- Live deployed URL: [add link here]
- GitHub repository: https://github.com/SajivJess/AI-Call-Center-Compliance---HCL
- Demo video (YouTube or Google Drive): [add link here]
- Optional slide deck: [add link here]
- README includes AI tools disclosure: yes

## Demo Video Requirements

- 2 to 5 minutes
- Screen recording with narration
- Show API auth, upload/test flow, and response output
- Demonstrate long-audio chunking and merged transcript behavior

## AI Tools Used

- OpenAI ChatGPT for implementation support, debugging, testing, and documentation assistance
- GitHub Copilot (GPT-5.4 mini) for implementation support, refactoring, testing, and documentation assistance

## Known Limitations

- Provider latency can increase response time.
- Accuracy depends on audio quality and language mixing.
- Long calls require chunking and multiple STT requests.
- SQLite is used for lightweight persistence, not as a horizontal-scale database.

## Tests

Run targeted checks:

```bash
pytest tests/test_api_contract.py tests/test_audio_and_language.py -q
```
