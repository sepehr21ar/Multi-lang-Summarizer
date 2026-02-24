import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cache_store import TTLCache
from config import get_settings
from file_utils import TextExtractionError, extract_text_from_file
from rate_limit import RateLimitMiddleware
from schemas import (
    AudioSummarizeResponse,
    AudioTranscribeResponse,
    LanguageEnum,
    SummarizeResponse,
    SummaryStyleEnum,
    TextSummarizeRequest,
)
from speech_service import SpeechService
from summarizer import SummarizationService

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
_VOICE_LANGUAGE_MAP = {
    LanguageEnum.english: "en",
    LanguageEnum.persian: "fa",
}
_SOURCE_LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "fa": "fa",
    "persian": "fa",
    "farsi": "fa",
}

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.requests_per_minute)

app.state.summarizer = SummarizationService(settings)
app.state.speech = SpeechService(settings)
app.state.cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


def _file_size_ok(file_bytes: bytes, max_mb: int) -> bool:
    return len(file_bytes) <= max_mb * 1024 * 1024


def _cache_key(*parts: str) -> str:
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _to_whisper_lang(language: LanguageEnum) -> str:
    return _VOICE_LANGUAGE_MAP[language]


def _ensure_voice_language_supported(language: LanguageEnum) -> None:
    if language not in {LanguageEnum.english, LanguageEnum.persian}:
        raise HTTPException(
            status_code=400,
            detail="Voice endpoints support only english and persian for output language.",
        )


def _normalize_source_language(source_language: str | None, language: LanguageEnum) -> str:
    if not source_language or not source_language.strip():
        return _to_whisper_lang(language)

    value = source_language.strip().lower()
    normalized = _SOURCE_LANGUAGE_ALIASES.get(value)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Invalid source_language. Use only: en/english or fa/persian/farsi.",
        )
    return normalized


def _get_cached_response(key: str):
    return app.state.cache.get(key)


def _transcribe_audio(
    file_bytes: bytes,
    file_name: str,
    source_language: str,
    return_timestamps: bool,
):
    try:
        return app.state.speech.transcribe(
            file_bytes=file_bytes,
            file_name=file_name,
            source_language=source_language,
            return_timestamps=return_timestamps,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {exc}") from exc


@app.get("/", include_in_schema=False)
def index():
    path = frontend_dir / "index.html"
    if path.exists():
        return FileResponse(path)
    return {"message": "Frontend not found. Use API docs at /docs"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "summarizer"}


@app.get("/ready")
def ready_check():
    return {
        "cohere_configured": app.state.summarizer.ready(),
        "speech_available": app.state.speech.ready(),
    }


@app.post("/v1/summarize/text", response_model=SummarizeResponse)
def summarize_text_api(payload: TextSummarizeRequest):
    text = payload.text.strip()
    if len(text) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail=f"Text is too large. Limit is {settings.max_text_chars} characters.")

    key = _cache_key("text", payload.language.value, payload.style.value, text)
    cached = _get_cached_response(key)
    if cached:
        return SummarizeResponse(**cached, cached=True)

    try:
        summary = app.state.summarizer.summarize_text(text, payload.language.value, payload.style.value)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Summarization failed: {exc}") from exc

    response_data = {
        "summary": summary,
        "language": payload.language,
        "style": payload.style,
        "original_length": len(text),
    }
    app.state.cache.set(key, response_data)
    return SummarizeResponse(**response_data)


@app.post("/v1/summarize/file", response_model=SummarizeResponse)
async def summarize_file_api(
    file: UploadFile = File(...),
    language: LanguageEnum = Form(LanguageEnum.english),
    style: SummaryStyleEnum = Form(SummaryStyleEnum.balanced),
):
    file_bytes = await file.read()

    if not _file_size_ok(file_bytes, settings.max_file_size_mb):
        raise HTTPException(status_code=413, detail=f"File is too large. Limit is {settings.max_file_size_mb} MB.")

    try:
        text = extract_text_from_file(file_bytes=file_bytes, file_name=file.filename or "uploaded_file")
    except TextExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    key = _cache_key("file", language.value, style.value, file.filename or "unknown", text)
    cached = _get_cached_response(key)
    if cached:
        return SummarizeResponse(**cached, cached=True)

    try:
        summary = app.state.summarizer.summarize_text(text, language.value, style.value)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Summarization failed: {exc}") from exc

    response_data = {
        "summary": summary,
        "language": language,
        "style": style,
        "original_length": len(text),
    }
    app.state.cache.set(key, response_data)
    return SummarizeResponse(**response_data)


@app.post("/v1/summarize/audio", response_model=AudioSummarizeResponse)
async def summarize_audio_api(
    file: UploadFile = File(...),
    language: LanguageEnum = Form(LanguageEnum.english),
    style: SummaryStyleEnum = Form(SummaryStyleEnum.balanced),
    source_language: str | None = Form(default=None),
    return_timestamps: bool = Form(default=False),
):
    file_bytes = await file.read()
    _ensure_voice_language_supported(language)
    normalized_source_language = _normalize_source_language(source_language, language)

    if not _file_size_ok(file_bytes, settings.max_audio_size_mb):
        raise HTTPException(status_code=413, detail=f"Audio is too large. Limit is {settings.max_audio_size_mb} MB.")

    transcription = _transcribe_audio(
        file_bytes=file_bytes,
        file_name=file.filename or "audio.wav",
        source_language=normalized_source_language,
        return_timestamps=return_timestamps,
    )

    transcript = transcription["transcript"]
    key = _cache_key("audio", language.value, style.value, transcript)
    cached = _get_cached_response(key)
    if cached:
        return AudioSummarizeResponse(**cached, cached=True)

    try:
        translated = app.state.summarizer.translate_text(transcript, language.value)
        summary = app.state.summarizer.summarize_text(translated, language.value, style.value)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Audio summarization failed: {exc}") from exc

    response_data = {
        "transcript": transcript,
        "translated_text": translated,
        "summary": summary,
        "language": language,
        "style": style,
        "language_detected": transcription["language_detected"],
        "duration_sec": transcription["duration_sec"],
        "segments": transcription["segments"],
    }
    app.state.cache.set(key, response_data)
    return AudioSummarizeResponse(**response_data)


@app.post("/v1/audio/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio_api(
    file: UploadFile = File(...),
    language: LanguageEnum = Form(LanguageEnum.english),
    source_language: str | None = Form(default=None),
    return_timestamps: bool = Form(default=False),
    translate_output: bool = Form(default=False),
):
    file_bytes = await file.read()
    _ensure_voice_language_supported(language)
    normalized_source_language = _normalize_source_language(source_language, language)

    if not _file_size_ok(file_bytes, settings.max_audio_size_mb):
        raise HTTPException(status_code=413, detail=f"Audio is too large. Limit is {settings.max_audio_size_mb} MB.")

    transcription = _transcribe_audio(
        file_bytes=file_bytes,
        file_name=file.filename or "audio.wav",
        source_language=normalized_source_language,
        return_timestamps=return_timestamps,
    )

    transcript = transcription["transcript"]
    key = _cache_key("audio_transcribe", language.value, str(translate_output), transcript)
    cached = _get_cached_response(key)
    if cached:
        return AudioTranscribeResponse(**cached, cached=True)

    output_text = None
    if translate_output:
        try:
            output_text = app.state.summarizer.translate_text(transcript, language.value)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Audio translation failed: {exc}") from exc

    response_data = {
        "transcript": transcript,
        "text_in_output_language": output_text,
        "translated": translate_output,
        "language": language,
        "language_detected": transcription["language_detected"],
        "duration_sec": transcription["duration_sec"],
        "segments": transcription["segments"],
    }
    app.state.cache.set(key, response_data)
    return AudioTranscribeResponse(**response_data)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = hashlib.md5(str(id(request)).encode()).hexdigest()[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
