# Multi-lang Summarizer (Production FastAPI + Frontend)

A production-ready multilingual summarization service with:
- Text summarization and translation
- File summarization (`pdf`, `docx`, `txt`, `md`, `csv`)
- Voice-to-text translation + summarization (`wav`, `mp3`, `m4a`, `flac`, `ogg`)
- Built-in frontend served by FastAPI
- Caching, rate limiting, health/readiness endpoints

## API Endpoints

- `GET /` frontend
- `GET /health` liveness
- `GET /ready` readiness
- `POST /v1/summarize/text`
- `POST /v1/summarize/file`
- `POST /v1/summarize/audio`
- `POST /v1/audio/transcribe` (voice-to-text tool, separate from summarizer)

OpenAPI docs: `http://localhost:8000/docs`

## Quick Start

1. Create virtual env and install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
# set COHERE_API_KEY in .env
```

3. Run API + frontend:

```bash
uvicorn app_api:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker compose up --build
```

## Voice Feature Notes

`/v1/summarize/audio` pipeline:
1. Transcribe audio with `faster-whisper`
2. Translate transcript to target language
3. Summarize translated text with selected style

If `faster-whisper` is missing, audio endpoint returns a clear error.

## Tests

```bash
pytest -q
```

## Legacy Gradio UI

`app_gradio.py` remains in the repo for legacy/demo usage. Production path is FastAPI (`app_api.py`) + built-in frontend (`frontend/index.html`).
