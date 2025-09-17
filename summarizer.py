from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_cohere import ChatCohere
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cohere.core.api_error import ApiError
from dotenv import load_dotenv
import os
import logging
import time
import tiktoken  
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """محاسبه تعداد توکن‌های متن"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

class AdvancedSummarizer:
    def __init__(self):
        self.llm = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the language model"""
        if not os.environ.get("COHERE_API_KEY"):
            api_key = input("لطفاً کلید API Cohere را وارد کنید: ").strip()
            if not api_key:
                raise ValueError("کلید API ضروری است")
            os.environ["COHERE_API_KEY"] = api_key
        
        # در initialize_llm تغییر دهید:
        self.llm = ChatCohere(
            model="command-a-03-2025",  # مدل بزرگ‌تر
            temperature=0.1,
            max_tokens=4000
        )

    def smart_text_split(self, text: str, max_tokens: int = 3500) -> List[Document]:
        """تقسیم هوشمندانه متن با در نظر گرفتن محدودیت توکن"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,  # افزایش chunk_size
            chunk_overlap=200,
            length_function=num_tokens_from_string,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    def summarize_large_text(self, text: str, language: str = "persian") -> str:
        """خلاصه‌سازی متن‌های بزرگ با استراتژی بهبود یافته"""
        try:
            if not text or not text.strip():
                return "❌ هیچ متنی برای خلاصه‌سازی وجود ندارد."
            
            # بررسی حجم متن
            total_tokens = num_tokens_from_string(text)
            logger.info(f"تعداد کل توکن‌ها: {total_tokens}")
            
            if total_tokens > 10000:  # اگر متن خیلی بزرگ است
                return "⚠️ متن بسیار بزرگ است. لطفاً متن کوتاه‌تری انتخاب کنید (حداکثر 8000 کلمه)."
            
            # تقسیم متن
            docs = self.smart_text_split(text)
            logger.info(f"تعداد بخش‌ها: {len(docs)}")
            
            if len(docs) == 1:
                # اگر متن در یک بخش جا می‌شود
                return self.summarize_directly(text, language)
            else:
                # برای متن‌های بزرگ‌تر از map_reduce استفاده کنید
                return self.summarize_map_reduce(docs, language)
                
        except Exception as e:
            logger.error(f"خطا در خلاصه‌سازی: {str(e)}")
            return f"⚠️ خطا در پردازش: {str(e)}"

    def summarize_directly(self, text: str, language: str) -> str:
        """خلاصه‌سازی مستقیم برای متن‌های کوچک"""
        prompt = ChatPromptTemplate.from_template(
            "متن زیر را به زبان {language} خلاصه کنید. خلاصه باید شامل نکات کلیدی، "
            "ایده‌های اصلی و نتیجه‌گیری باشد:\n\n{text}"
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"text": text, "language": language})
        return result.content

    def summarize_map_reduce(self, docs: List[Document], language: str) -> str:
        """خلاصه‌سازی با الگوی map-reduce برای متن‌های بزرگ"""
        map_prompt_template = ChatPromptTemplate.from_template(
            "بخش زیر از یک متن بزرگ را به زبان {language} خلاصه کنید. "
            "روی نکات کلیدی و اطلاعات مهم تمرکز کنید:\n\n{text}"
        )

        combine_prompt_template = ChatPromptTemplate.from_template(
            "خلاصه‌های زیر از بخش‌های مختلف یک متن هستند. آن‌ها را به یک خلاصه منسجم "
            "و جامع به زبان {language} ترکیب کنید. ساختار منطقی داشته باشد و "
            "همه نکات مهم را پوشش دهد:\n\n{text}"
        )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=True
        )

        result = chain.invoke({
            "input_documents": docs, 
            "language": language
        })
        
        return result["output_text"]

# نمونه اصلی
summarizer = AdvancedSummarizer()

def summarize_text(text: str, language: str = "persian") -> str:
    """تابع wrapper برای سازگاری"""
    return summarizer.summarize_large_text(text, language)