from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum

from summarizer import summarize_text
from file_utils import extract_text_from_file


class LanguageEnum(str, Enum):
    english = "english"
    persian = "persian"
    french = "french"
    german = "german"
    spanish = "spanish"
    arabic = "arabic"

app = FastAPI(
    title="Multilingual Summarization API",
    description="Translation + Summarization API using Cohere & LangChain",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeResponse(BaseModel):
    summary: str
    language: LanguageEnum
    original_length: int

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "summarizer"
    }

@app.post("/summarize/text", response_model=SummarizeResponse)
def summarize_text_api(
    text: str = Form(..., description="متن برای خلاصه‌سازی"),
    language: LanguageEnum = Form(LanguageEnum.persian, description="زبان خروجی")
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    summary = summarize_text(
        text,
        language=language.value  
    )

    return {
        "summary": summary,
        "language": language,
        "original_length": len(text)
    }

@app.post("/summarize/file", response_model=SummarizeResponse)
async def summarize_file_api(
    file: UploadFile = File(..., description="فایل ورودی (PDF, DOCX, TXT, CSV, MD)"),
    language: LanguageEnum = Form(LanguageEnum.persian, description="زبان خروجی")
):
    file_bytes = await file.read()

    extracted_text = extract_text_from_file(
        file_bytes=file_bytes,
        file_name=file.filename
    )

    if extracted_text.startswith("⚠️") or extracted_text.startswith("❌"):
        raise HTTPException(status_code=400, detail=extracted_text)

    summary = summarize_text(
        extracted_text,
        language=language.value
    )

    return {
        "summary": summary,
        "language": language,
        "original_length": len(extracted_text)
    }
