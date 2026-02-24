import io

from fastapi.testclient import TestClient

from app_api import app


class FakeSummarizer:
    def ready(self):
        return True

    def summarize_text(self, text: str, language: str = "english", style: str = "balanced"):
        return f"summary[{language}/{style}] {text[:20]}"

    def translate_text(self, text: str, target_language: str):
        return f"translated[{target_language}] {text}"


class FakeSpeech:
    def ready(self):
        return True

    def transcribe(self, file_bytes: bytes, file_name: str, source_language=None, return_timestamps=False):
        segments = [{"start": 0.0, "end": 1.0, "text": "hello world"}] if return_timestamps else []
        return {
            "transcript": "hello world",
            "language_detected": source_language or "en",
            "duration_sec": 1.0,
            "segments": segments,
        }


def _client() -> TestClient:
    app.state.summarizer = FakeSummarizer()
    app.state.speech = FakeSpeech()
    return TestClient(app)


def test_health():
    client = _client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_text_summarize():
    client = _client()
    r = client.post(
        "/v1/summarize/text",
        json={"text": "this is a test", "language": "english", "style": "balanced"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert body["language"] == "english"


def test_file_summarize():
    client = _client()
    files = {"file": ("sample.txt", io.BytesIO(b"some content to summarize"), "text/plain")}
    data = {"language": "english", "style": "concise"}
    r = client.post("/v1/summarize/file", files=files, data=data)
    assert r.status_code == 200
    assert "summary" in r.json()


def test_audio_summarize():
    client = _client()
    files = {"file": ("sample.wav", io.BytesIO(b"fake audio bytes"), "audio/wav")}
    data = {"language": "english", "style": "balanced", "return_timestamps": "true"}
    r = client.post("/v1/summarize/audio", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert payload["transcript"] == "hello world"
    assert "summary" in payload


def test_audio_transcribe_only():
    client = _client()
    files = {"file": ("sample.wav", io.BytesIO(b"fake audio bytes"), "audio/wav")}
    data = {"language": "persian", "return_timestamps": "true"}
    r = client.post("/v1/audio/transcribe", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert payload["transcript"] == "hello world"
    assert payload["text_in_output_language"] is None
    assert payload["translated"] is False


def test_audio_transcribe_with_translation():
    client = _client()
    files = {"file": ("sample.wav", io.BytesIO(b"fake audio bytes"), "audio/wav")}
    data = {"language": "persian", "translate_output": "true"}
    r = client.post("/v1/audio/transcribe", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert payload["transcript"] == "hello world"
    assert payload["translated"] is True
    assert payload["text_in_output_language"].startswith("translated[persian]")
