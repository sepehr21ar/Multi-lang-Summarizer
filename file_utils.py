from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader, PyPDFLoader
import tempfile
import os
import logging

# 💡 حذف وابستگی‌های OCR: pytesseract و pdf2image دیگر لازم نیستند.

logging.basicConfig(level=logging.INFO)
# استفاده از __name__ یک روش بهتر برای نام‌گذاری logger است.
logger = logging.getLogger(__name__) 

def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """Extract text from a file using LangChain loaders where possible."""
    if not file_bytes:
        return "⚠️ فایل خالی است."

    fname = file_name.lower()
    suffix = "." + fname.split(".")[-1] if "." in fname else ""

    loader_map = {
        ".pdf": lambda path: PyPDFLoader(path),
        ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
        ".docx": lambda path: Docx2txtLoader(path),
        ".txt": lambda path: TextLoader(path, encoding="utf-8"),
        ".md": lambda path: TextLoader(path, encoding="utf-8"),
    }

    if suffix in loader_map:
        # برای لودرها، فایل موقت ایجاد می‌کنیم.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader_fn = loader_map[suffix]
            loader = loader_fn(tmp_path)
            docs = loader.load()
            
            # 💡 بهبود: حذف کامل بخش OCR برای PDF
            if not docs:
                if suffix == ".pdf":
                     # اگر PyPDFLoader متنی پیدا نکرد، فرض می‌کنیم PDF اسکن شده است.
                     return "⚠️ فایل PDF متنی نیست یا خالی است. پشتیبانی از PDF اسکن شده (OCR) غیرفعال است."
                else:
                    return "⚠️ هیچ متنی پیدا نشد."
                    
            contents = [d.page_content.strip() for d in docs if d.page_content.strip()]
            return "\n\n".join(contents)
            
        except Exception as e:
            logger.error(f"خطا در پردازش فایل {suffix}: {str(e)}")
            return f"⚠️ خطا در پردازش فایل: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # fallback برای فایل‌های دیگر
    try:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1256']
        for encoding in encodings:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"خطا در رمزگشایی فایل: {str(e)}"