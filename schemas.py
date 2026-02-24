from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class LanguageEnum(str, Enum):
    english = "english"
    persian = "persian"
    french = "french"
    german = "german"
    spanish = "spanish"
    arabic = "arabic"


class SummaryStyleEnum(str, Enum):
    concise = "concise"
    balanced = "balanced"
    bullet_points = "bullet_points"


class ErrorResponse(BaseModel):
    detail: str


class TextSummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: LanguageEnum = LanguageEnum.english
    style: SummaryStyleEnum = SummaryStyleEnum.balanced


class SummarizeResponse(BaseModel):
    summary: str
    language: LanguageEnum
    style: SummaryStyleEnum
    original_length: int
    cached: bool = False


class AudioSegment(BaseModel):
    start: float
    end: float
    text: str


class AudioSummarizeResponse(BaseModel):
    transcript: str
    translated_text: str
    summary: str
    language: LanguageEnum
    style: SummaryStyleEnum
    language_detected: str
    duration_sec: float
    cached: bool = False
    segments: List[AudioSegment] = Field(default_factory=list)


class AudioTranscribeResponse(BaseModel):
    transcript: str
    text_in_output_language: str | None = None
    translated: bool = False
    language: LanguageEnum
    language_detected: str
    duration_sec: float
    cached: bool = False
    segments: List[AudioSegment] = Field(default_factory=list)
