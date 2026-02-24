import gradio as gr
from summarizer import summarize_text
from file_utils import TextExtractionError, extract_text_from_file # 💡 اطمینان از وارد کردن صحیح 'utils'
import os
import time

# تنظیمات پروکسی اگر نیاز باشد (مانند نمونه اصلی شما)
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
# os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# نگاشت زبان‌ها برای نمایش بهتر
LANGUAGE_DISPLAY = {
    "english": "🇺🇸 English",
    "persian": "🇮🇷 فارسی",
    "french": "🇫🇷 Français",
    "german": "🇩🇪 Deutsch",
    "spanish": "🇪🇸 Español",
    "arabic": "🇸🇦 العربية"
}
LANGUAGE_VALUES = {
    "🇺🇸 English": "english",
    "🇮🇷 فارسی": "persian",
    "🇫🇷 Français": "french",
    "🇩🇪 Deutsch": "german",
    "🇪🇸 Español": "spanish",
    "🇸🇦 العربية": "arabic"
}
def process_input(file_path, text_input, language_display, progress=gr.Progress()):
    extracted_text = ""

    language = LANGUAGE_VALUES.get(language_display, "persian")

    progress(0.1, desc="در حال بررسی ورودی...")

    if file_path:
        progress(0.3, desc="در حال استخراج متن از فایل...")

        # خواندن بایت‌ها
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_name = os.path.basename(file_path)

        try:
            extracted_text = extract_text_from_file(file_bytes, file_name)
        except TextExtractionError as exc:
            return f"⚠️ {exc}", ""

    elif text_input and text_input.strip():
        extracted_text = text_input.strip()

    else:
        return "❌ لطفاً فایل یا متن وارد کنید.", ""

    # بررسی خطا
    if not extracted_text or extracted_text.startswith("⚠️") or extracted_text.startswith("❌"):
        return extracted_text, extracted_text

    try:
        progress(0.6, desc="در حال خلاصه‌سازی...")
        summary = summarize_text(extracted_text, language=language)

        progress(0.9, desc="در حال آماده‌سازی نتیجه...")
        time.sleep(0.5)

        return summary.strip(), extracted_text

    except Exception as e:
        return f"⚠️ خطا در پردازش: {str(e)}", extracted_text

# تم Persian-friendly
css = """
.gradio-container {
    font-family: Vazirmatn, Tahoma, sans-serif;
}
h1 {
    text-align: center;
    color: #2c5aa0;
}
"""

with gr.Blocks(css=css, title="خلاصه‌ساز هوشمند") as demo:
    gr.Markdown("""
    # 🌍 خلاصه‌ساز چندزبانه هوشمند
    متن یا فایل خود را وارد کنید و خلاصه‌ای حرفه‌ای دریافت کنید
    
    **پشتیبانی از فایل‌ها:** PDF, Word, Text, CSV, Markdown""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 ورودی")
            
            file_input = gr.File(
    label="آپلود فایل",
    file_types=[".pdf", ".docx", ".csv", ".txt", ".md"],
    type="filepath"
)

            
            text_input = gr.Textbox(
                label="یا متن خود را وارد کنید",
                lines=8,
                placeholder="متن خود را اینجا بنویسید یا جای‌گذاری کنید...",
                max_lines=20
            )
            
            lang_dropdown = gr.Dropdown(
                label="زبان خلاصه",
                choices=list(LANGUAGE_DISPLAY.values()),
                value=LANGUAGE_DISPLAY["persian"],
                info="زبان خروجی خلاصه"
            )
            
            summarize_btn = gr.Button(
                "✨ شروع خلاصه‌سازی", 
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 خروجی")
            
            with gr.Tab("خلاصه"):
                output_summary = gr.Textbox(
                    label="خلاصه تولید شده",
                    lines=10,
                    interactive=False,
                    show_copy_button=True
                )
            
            with gr.Tab("متن استخراج شده"):
                extracted_text_display = gr.Textbox(
                    label="متن اصلی استخراج شده",
                    lines=10,
                    interactive=False,
                    show_copy_button=True
                )
    
    # وضعیت سیستم
    status = gr.Textbox(
        label="وضعیت",
        value="✅ آماده به کار",
        interactive=False
    )
    
    # رویدادها
    summarize_btn.click(
        fn=process_input,
        inputs=[file_input, text_input, lang_dropdown],
        outputs=[output_summary, extracted_text_display],
        show_progress="full"
    )

    # رویدادهای clear: با تغییر ورودی‌ها، خروجی‌ها پاک شوند
    file_input.change(
        lambda: ("", ""),
        outputs=[output_summary, extracted_text_display],
        # اطمینان از اینکه اگر فقط متن وجود دارد، فایل آن را پاک نکند.
        # اما در اینجا برای سادگی، هر تغییر هر دو خروجی را پاک می‌کند.
        
    )

    text_input.change(
        lambda: ("", ""),
        outputs=[output_summary, extracted_text_display]
    )

# راه‌اندازی سرور
if __name__ == "__main__":
    try:
        import os

        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"

        demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_api=False,
    inbrowser=True  
)





    finally:
        print("Done")
