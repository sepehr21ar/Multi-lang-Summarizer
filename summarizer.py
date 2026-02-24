import logging
from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import Settings

logger = logging.getLogger(__name__)

_STYLE_RULES = {
    "concise": "Keep it very short and direct in 3-5 sentences.",
    "balanced": "Cover the key points and the main conclusion clearly.",
    "bullet_points": "Return the summary as clear bullet points."
}


class SummarizationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm: ChatCohere | None = None
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def ready(self) -> bool:
        return bool(self.settings.cohere_api_key)

    def _get_llm(self) -> ChatCohere:
        if not self.settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not configured.")

        if self._llm is None:
            self._llm = ChatCohere(
                cohere_api_key=self.settings.cohere_api_key,
                model=self.settings.cohere_model,
                temperature=self.settings.cohere_temperature,
                max_tokens=self.settings.cohere_max_tokens,
            )
        return self._llm

    def _token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _split(self, text: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2200,
            chunk_overlap=320,
            length_function=self._token_count,
            separators=["\n\n", "\n", " ", ""],
        )
        return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    def _summarize_chunk(self, text: str, language: str, style: str, chunked: bool = False) -> str:
        llm = self._get_llm()
        context_rule = "This is one part of a larger text." if chunked else ""
        style_rule = _STYLE_RULES.get(style, _STYLE_RULES["balanced"])

        prompt = ChatPromptTemplate.from_template(
            """
Summarize the following text in {language}.
{context_rule}
Style requirement: {style_rule}
Preserve critical names, numbers, and claims.

Text:
{text}
""".strip()
        )

        chain = prompt | llm
        result = chain.invoke(
            {
                "text": text,
                "language": language,
                "style_rule": style_rule,
                "context_rule": context_rule,
            }
        )
        return result.content.strip()

    def summarize_text(self, text: str, language: str = "english", style: str = "balanced") -> str:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        text = text.strip()
        docs = self._split(text)

        if len(docs) == 1:
            return self._summarize_chunk(docs[0].page_content, language, style, chunked=False)

        partials = [self._summarize_chunk(doc.page_content, language, style, chunked=True) for doc in docs]

        combined = "\n\n".join(partials)
        rounds = 0
        while self._token_count(combined) > 3500 and rounds < 3:
            sub_docs = self._split(combined)
            partials = [self._summarize_chunk(doc.page_content, language, style, chunked=True) for doc in sub_docs]
            combined = "\n\n".join(partials)
            rounds += 1

        return self._summarize_chunk(combined, language, style, chunked=False)

    def translate_text(self, text: str, target_language: str) -> str:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_template(
            """
Translate the following text into {target_language}.
Keep meaning and named entities accurate. Do not summarize.

Text:
{text}
""".strip()
        )
        chain = prompt | llm
        result = chain.invoke({"text": text, "target_language": target_language})
        return result.content.strip()


__all__ = ["SummarizationService"]


_default_service: SummarizationService | None = None


def _get_default_service() -> SummarizationService:
    global _default_service
    if _default_service is None:
        from config import get_settings

        _default_service = SummarizationService(get_settings())
    return _default_service


def summarize_text(text: str, language: str = "english", style: str = "balanced") -> str:
    return _get_default_service().summarize_text(text, language, style)
