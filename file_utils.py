from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader, PyPDFLoader
import tempfile


def extract_text_from_file (file_bytes: bytes, file_name : str) -> str:
       """Extract text from a file using LangChain loaders where possible."""
       fname = file_name.lower()
       text  = ""
       suffix = "." + fname.split(".")[-1] if "." in fname else ""
       loader_map = {
        ".pdf": lambda path: PyPDFLoader(path),
        ".csv": lambda path: CSVLoader(path),
        ".docx": lambda path: Docx2txtLoader(path),
        ".txt": lambda path: TextLoader(path, encoding="utf-8"),
        ".md": lambda path: TextLoader(path, encoding="utf-8"),
    }
       
       if suffix in loader_map:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            loader = loader_map[suffix](tmp.name)
            docs = loader.load()
            return "\n\n".join(d.page_content for d in docs)

       try:
              return file_bytes.decode("utf-8", errors="ignore")
       except Exception:
              return ""