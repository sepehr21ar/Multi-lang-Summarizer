import os
import tempfile
from typing import Any
import logging
from pathlib import Path
import shutil

from config import Settings

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency runtime
    WhisperModel = None

logger = logging.getLogger(__name__)
_SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class SpeechService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[str, WhisperModel] = {}

    def ready(self) -> bool:
        return WhisperModel is not None

    def _resolve_model_name(self, source_language: str | None) -> str:
        if source_language == "fa":
            return self.settings.whisper_model_size_persian
        return self.settings.whisper_model_size

    def _prepare_local_model_dir(self, model_name: str) -> str:
        model_path = Path(model_name)
        if not model_path.exists() or not model_path.is_dir():
            return model_name

        config_file = model_path / "config.json"
        if not config_file.exists():
            candidates = sorted(model_path.glob("config*.json"))
            if candidates:
                shutil.copy2(candidates[0], config_file)
                logger.warning(
                    "Local Whisper model had no config.json. Copied %s to config.json.",
                    candidates[0].name,
                )

        required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        missing = [name for name in required if not (model_path / name).exists()]
        if missing:
            raise RuntimeError(
                "Local Whisper model directory is incomplete. Missing files: "
                + ", ".join(missing)
            )

        return str(model_path)

    def _get_model(self, model_name: str):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed. Install it to enable audio transcription.")

        prepared_model_name = self._prepare_local_model_dir(model_name)
        if prepared_model_name not in self._models:
            try:
                self._models[prepared_model_name] = WhisperModel(
                    prepared_model_name,
                    compute_type=self.settings.whisper_compute_type,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Whisper model '{prepared_model_name}'. Check internet/proxy/firewall, then retry. "
                    "If needed, pre-download with: "
                    "python -c \"from faster_whisper import WhisperModel; WhisperModel('tiny', compute_type='int8')\""
                ) from exc
        return self._models[prepared_model_name]

    def transcribe(
        self,
        file_bytes: bytes,
        file_name: str,
        source_language: str | None = None,
        return_timestamps: bool = False,
    ) -> dict[str, Any]:
        if not file_bytes:
            raise ValueError("Audio file is empty.")

        suffix = os.path.splitext(file_name.lower())[1] or ".wav"
        if suffix not in _SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format: {suffix}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.debug("Transcribing audio file with suffix %s", suffix)
            model_name = self._resolve_model_name(source_language)
            model = self._get_model(model_name)
            try:
                segments, info = model.transcribe(
                    tmp_path,
                    language=source_language,
                    task="transcribe",
                    beam_size=self.settings.whisper_beam_size,
                    best_of=self.settings.whisper_best_of,
                    temperature=self.settings.whisper_temperature,
                    vad_filter=self.settings.whisper_vad_filter,
                    condition_on_previous_text=self.settings.whisper_condition_on_previous_text,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Audio decoding/transcription failed. Ensure ffmpeg is installed and the model is downloaded."
                ) from exc

            transcript_parts = []
            ts_segments = []
            duration = 0.0

            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)
                duration = float(segment.end)
                if return_timestamps:
                    ts_segments.append(
                        {
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "text": text,
                        }
                    )

            transcript = " ".join(transcript_parts).strip()
            if not transcript:
                raise ValueError("No speech detected in audio file.")

            detected = getattr(info, "language", None) or source_language or "unknown"

            return {
                "transcript": transcript,
                "language_detected": detected,
                "duration_sec": duration,
                "segments": ts_segments,
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
