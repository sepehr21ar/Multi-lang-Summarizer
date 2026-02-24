import logging
import os
import tempfile

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)


class TextExtractionError(Exception):
    pass


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".docx", ".txt", ".md"}


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    if not file_bytes:
        raise TextExtractionError("Uploaded file is empty.")

    suffix = os.path.splitext(file_name.lower())[1]
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise TextExtractionError(f"Unsupported file type: {suffix}")

    if not suffix:
        return _decode_text(file_bytes)

    loader_map = {
        ".pdf": lambda p: PyPDFLoader(p),
        ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
        ".docx": lambda p: Docx2txtLoader(p),
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".md": lambda p: TextLoader(p, encoding="utf-8"),
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = loader_map[suffix](tmp_path)
        docs = loader.load()
        if not docs:
            if suffix == ".pdf":
                raise TextExtractionError(
                    "No extractable text found in PDF. It may be scanned and requires OCR support."
                )
            raise TextExtractionError("No text could be extracted from the file.")

        chunks = [doc.page_content.strip() for doc in docs if doc.page_content and doc.page_content.strip()]
        content = "\n\n".join(chunks).strip()
        if not content:
            raise TextExtractionError("File was parsed but no readable text content was found.")
        return content
    except TextExtractionError:
        raise
    except Exception as exc:
        logger.exception("File extraction failed for %s", suffix)
        raise TextExtractionError(f"Failed to parse file: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
