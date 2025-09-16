from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_cohere import ChatCohere
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


def summarize_text(text : str, language:str = "english") -> str:
       """Summarize long text into the target language using modern techniques."""

       if not text.strip():
              return "❌ No text to summarize."
       splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
       docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

       llm = ChatCohere(model="command-r-plus", temperature=0.09)

       map_prompt_template = ChatPromptTemplate.from_template(
       "Summarize the following text in {language}:\n\n{text}\n\n"
       "Focus only on the key points and main ideas."
       )

       combine_prompt_template = ChatPromptTemplate.from_template(
       "You are given several partial summaries. Combine them into a single, "
       "well-structured summary in {language}:\n\n{text}\n\n"
       "Make sure the summary is coherent and highlight the most important insights.")

       chain = load_summarize_chain(
              llm,
              chain_type="map_reduce",
              map_prompt=map_prompt_template,
              combine_prompt=combine_prompt_template
       )
       result: Dict[str, Any] = chain.invoke({"input_documents": docs, "language": language})
       return result["output_text"]
      