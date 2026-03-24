import base64
import ast
import os
import sys
from html import escape
from pathlib import Path
from typing import List

import gradio as gr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = r"D:\ecommerce-multimodal-rag"
SRC_ROOT = rf"{PROJECT_ROOT}\src"
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from local_llm import warmup_local_llm
from pipeline import search
from query_utils import contains_chinese, warmup_translation
from rag_answer import generate_answer, generate_quick_summary
from retrieve import warmup_retrieval_models


CUSTOM_CSS = """
:root {
  --page-bg: #f7f3ef;
  --panel-bg: rgba(255, 252, 248, 0.96);
  --panel-border: #ddd2c7;
  --text-main: #352b26;
  --text-soft: #695b54;
  --accent: #c9785d;
  --accent-strong: #b9634a;
  --accent-track: #ead9cf;
  --cream: #fffaf6;
  --input-border: #d8c8bb;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, #fff8f3 0%, rgba(255, 248, 243, 0) 34%),
    radial-gradient(circle at top right, #f1e5db 0%, rgba(241, 229, 219, 0) 28%),
    linear-gradient(180deg, #faf7f2 0%, #f4eee8 100%);
  color: var(--text-main);
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.app-shell {
  max-width: 1280px;
  margin: 0 auto;
}

.hero-card {
  background: linear-gradient(135deg, rgba(255, 250, 246, 0.98), rgba(244, 233, 224, 0.94));
  border: 1px solid rgba(211, 194, 179, 0.9);
  border-radius: 28px;
  padding: 28px 30px 24px;
  box-shadow: 0 18px 60px rgba(145, 120, 101, 0.12);
  margin-bottom: 18px;
}

.hero-kicker {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: #f2e6dc;
  color: #7f5f51;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-title {
  margin: 14px 0 10px;
  font-size: 40px;
  line-height: 1.08;
  font-weight: 700;
  color: #3f332d;
}

.hero-subtitle {
  margin: 0;
  max-width: 760px;
  color: var(--text-soft);
  font-size: 17px;
  line-height: 1.8;
}

.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .gr-panel,
.gradio-container .gr-form {
  background: var(--panel-bg) !important;
  border: 1px solid var(--panel-border) !important;
  box-shadow: 0 10px 30px rgba(139, 117, 100, 0.08);
  border-radius: 24px !important;
}

.gradio-container label,
.gradio-container .gr-form .label-wrap label,
.gradio-container .block_label {
  color: var(--text-main) !important;
  font-weight: 700 !important;
  font-size: 16px !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-textbox,
.gradio-container .wrap,
.gradio-container .gr-dataframe {
  background: var(--cream) !important;
  color: var(--text-main) !important;
  border-color: var(--input-border) !important;
}

.gradio-container input,
.gradio-container textarea {
  font-size: 17px !important;
  line-height: 1.7 !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #8a7b72 !important;
  opacity: 1 !important;
  font-size: 16px !important;
}

.gradio-container .gr-dataframe table,
.gradio-container .gr-dataframe td,
.gradio-container .gr-dataframe th {
  color: var(--text-main) !important;
  font-size: 15px !important;
}

.gradio-container .gr-dataframe th {
  background: linear-gradient(180deg, #efe1d6 0%, #e6d5c8 100%) !important;
  color: #4a3d37 !important;
  font-weight: 700 !important;
}

.gradio-container .gr-dataframe td {
  background: #fffaf6 !important;
}

#status-box textarea,
#status-box input {
  background: linear-gradient(135deg, #f4e7dc 0%, #f0dfd1 100%) !important;
  color: #4b3d36 !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  border: 1px solid #d9c4b5 !important;
}

#search-btn {
  background: linear-gradient(135deg, var(--accent) 0%, #d98d6f 100%) !important;
  color: white !important;
  border: none !important;
  font-weight: 700 !important;
  font-size: 17px !important;
  box-shadow: 0 12px 24px rgba(201, 120, 93, 0.28);
}

#search-btn:hover {
  background: linear-gradient(135deg, var(--accent-strong) 0%, #cc7b5d 100%) !important;
  transform: translateY(-1px);
}

#detail-btn {
  background: linear-gradient(135deg, #e6ddd5 0%, #d7cbc2 100%) !important;
  color: #4e413b !important;
  border: 1px solid #cfbeb2 !important;
  font-weight: 600 !important;
  font-size: 16px !important;
}

#search-btn, #detail-btn {
  min-height: 54px;
  border-radius: 16px !important;
  transition: all 0.2s ease;
}

.gradio-container .gr-gallery,
.gradio-container .gr-dataframe,
.gradio-container .gr-image,
.gradio-container .gr-textbox,
#product-cards {
  border-radius: 20px !important;
}

.gradio-container input[type="range"] {
  accent-color: var(--accent) !important;
}

.gradio-container .gr-slider input[type="range"]::-webkit-slider-runnable-track {
  background: linear-gradient(90deg, var(--accent-track) 0%, #e5c7b8 100%) !important;
  height: 8px !important;
  border-radius: 999px !important;
}

.gradio-container .gr-slider input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none !important;
  width: 20px !important;
  height: 20px !important;
  margin-top: -6px !important;
  border-radius: 50% !important;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
  border: 2px solid #fff8f3 !important;
  box-shadow: 0 4px 10px rgba(185, 99, 74, 0.25) !important;
}

.gradio-container .gr-slider input[type="range"]::-moz-range-track {
  background: linear-gradient(90deg, var(--accent-track) 0%, #e5c7b8 100%) !important;
  height: 8px !important;
  border-radius: 999px !important;
}

.gradio-container .gr-slider input[type="range"]::-moz-range-thumb {
  width: 20px !important;
  height: 20px !important;
  border-radius: 50% !important;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
  border: 2px solid #fff8f3 !important;
  box-shadow: 0 4px 10px rgba(185, 99, 74, 0.25) !important;
}

.section-hint {
  color: #6f625a;
  font-size: 15px;
  line-height: 1.7;
  margin-top: -2px;
  margin-bottom: 12px;
}

.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 18px;
}

.product-card {
  background: linear-gradient(180deg, #fffaf6 0%, #f6ece4 100%);
  border: 1px solid #dcc8ba;
  border-radius: 22px;
  overflow: hidden;
  box-shadow: 0 12px 26px rgba(129, 104, 88, 0.12);
}

.product-card-image {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  display: block;
  background: #f7efe8;
}

.product-card-body {
  padding: 14px 14px 16px;
  position: relative;
}

.product-card-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
  padding: 6px 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, #c9785d 0%, #db9b7e 100%);
  color: #fffaf6;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
  box-shadow: 0 8px 16px rgba(185, 99, 74, 0.24);
}

.product-card-title {
  color: #3f332d;
  font-size: 17px;
  line-height: 1.45;
  font-weight: 700;
  margin-bottom: 10px;
  min-height: 52px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.product-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.product-card-tag {
  display: inline-flex;
  align-items: center;
  padding: 6px 11px;
  border-radius: 999px;
  background: linear-gradient(135deg, #f3e2d7 0%, #ead6ca 100%);
  color: #704f42;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid #e0c8ba;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

.product-card-selling {
  color: #5b4d46;
  font-size: 14px;
  line-height: 1.65;
  min-height: 46px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.product-card-score {
  margin-top: 12px;
  color: #b9634a;
  font-size: 13px;
  font-weight: 700;
}

.product-empty {
  padding: 24px;
  border-radius: 20px;
  background: linear-gradient(180deg, #fffaf6 0%, #f6ece4 100%);
  border: 1px solid #dcc8ba;
  color: #685a52;
  font-size: 15px;
}
"""


def image_to_data_uri(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower().replace(".", "") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    with open(image_path, "rb") as file_obj:
        encoded = base64.b64encode(file_obj.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def parse_multilingual_value(raw_value):
    if not isinstance(raw_value, str):
        return raw_value
    text = raw_value.strip()
    if not text.startswith("[") or "'language_tag'" not in text:
        return raw_value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return raw_value


def choose_display_text(raw_value, prefer_chinese: bool) -> str:
    parsed = parse_multilingual_value(raw_value)
    if isinstance(parsed, list):
        if prefer_chinese:
            preferred_prefixes = ["zh", "zh_cn", "zh_tw", "zh_hk", "en"]
        else:
            preferred_prefixes = ["en", "en_gb", "en_us", "en_ae"]

        normalized = []
        for item in parsed:
            if isinstance(item, dict):
                tag = str(item.get("language_tag", "")).lower()
                value = str(item.get("value", "")).strip()
                if value:
                    normalized.append((tag, value))

        for prefix in preferred_prefixes:
            for tag, value in normalized:
                if tag.startswith(prefix):
                    return value
        if normalized:
            return normalized[0][1]
    return str(raw_value or "").strip()


def localize_display_text(text: str, prefer_chinese: bool) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if prefer_chinese:
        if text.lower() == "unknown":
            return "未知"
        return text
    return text


def compact_text(text: str, max_len: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def build_selling_points(item: dict, prefer_chinese: bool) -> str:
    parts = []
    color = localize_display_text(choose_display_text(item.get("color", ""), prefer_chinese), prefer_chinese)
    material = localize_display_text(choose_display_text(item.get("material", ""), prefer_chinese), prefer_chinese)
    style = localize_display_text(choose_display_text(item.get("style", ""), prefer_chinese), prefer_chinese)
    if color:
        parts.append(("颜色" if prefer_chinese else "Color") + f": {color}")
    if material:
        parts.append(("材质" if prefer_chinese else "Material") + f": {material}")
    if style:
        parts.append(("风格" if prefer_chinese else "Style") + f": {style}")
    text = " · ".join(parts)
    if text:
        return compact_text(text, 78)
    description = localize_display_text(choose_display_text(item.get("description", ""), prefer_chinese), prefer_chinese)
    return compact_text(description, 88)


def localize_results_for_display(results: List[dict], prefer_chinese: bool) -> List[dict]:
    localized = []
    for item in results:
        updated = dict(item)
        updated["title"] = localize_display_text(choose_display_text(item.get("title", ""), prefer_chinese), prefer_chinese)
        updated["brand"] = localize_display_text(choose_display_text(item.get("brand", ""), prefer_chinese), prefer_chinese)
        updated["product_type"] = localize_display_text(choose_display_text(item.get("product_type", ""), prefer_chinese), prefer_chinese)
        updated["color"] = localize_display_text(choose_display_text(item.get("color", ""), prefer_chinese), prefer_chinese)
        updated["material"] = localize_display_text(choose_display_text(item.get("material", ""), prefer_chinese), prefer_chinese)
        updated["style"] = localize_display_text(choose_display_text(item.get("style", ""), prefer_chinese), prefer_chinese)
        updated["description"] = localize_display_text(choose_display_text(item.get("description", ""), prefer_chinese), prefer_chinese)
        localized.append(updated)
    return localized


def render_product_cards(results: List[dict], query_text: str) -> str:
    if not results:
        return "<div class='product-empty'>No products found yet. Try a text query or upload an image.</div>"

    prefer_chinese = contains_chinese(query_text or "")
    results = localize_results_for_display(results, prefer_chinese)
    cards = []
    for index, item in enumerate(results):
        image_src = image_to_data_uri(item["image_path"])
        title = escape(compact_text(item.get("title", ""), 72))
        brand = escape(compact_text(item.get("brand", ""), 24))
        product_type = escape(compact_text(item.get("product_type", ""), 24))
        selling = escape(build_selling_points(item, prefer_chinese))
        score = float(item.get("score", 0.0))
        badge = ""
        if index == 0:
            badge_text = "Best Match" if not prefer_chinese else "最佳推荐"
            badge = f"<div class='product-card-badge'>{badge_text}</div>"
        cards.append(
            f"""
            <div class="product-card">
              <img class="product-card-image" src="{image_src}" alt="{title}">
              <div class="product-card-body">
                {badge}
                <div class="product-card-title">{title}</div>
                <div class="product-card-tags">
                  <span class="product-card-tag">{brand}</span>
                  <span class="product-card-tag">{product_type}</span>
                </div>
                <div class="product-card-selling">{selling}</div>
                <div class="product-card-score">Match Score: {score:.4f}</div>
              </div>
            </div>
            """
        )
    return f"<div class='product-grid'>{''.join(cards)}</div>"


def format_table(results: List[dict]):
    table = []
    for item in results:
        table.append([item["product_id"], item["title"], item["brand"], item["product_type"], item["score"]])
    return table


def run_search(query_text, query_image, top_k):
    image_path = query_image if isinstance(query_image, str) else None
    results = search(query_text=query_text or None, query_image=image_path, top_k=int(top_k))
    quick_summary = generate_quick_summary(query_text or "Describe the best matching products.", results)
    detail_placeholder = (
        "点击 Generate Detailed Answer 生成详细回答。"
        if contains_chinese(query_text or "")
        else "Click Generate Detailed Answer to generate a detailed answer."
    )
    return render_product_cards(results, query_text or ""), format_table(results), quick_summary, detail_placeholder, results


def run_detailed_answer(query_text, results):
    results = results or []
    if not results:
        return "请先执行 Search。" if contains_chinese(query_text or "") else "Please run Search first."
    return generate_answer(query_text or "Describe the best matching products.", results)


def warmup_runtime():
    warmup_retrieval_models()
    warmup_local_llm()
    warmup_translation()
    return "Runtime warmed up: retrieval, RAG, and Chinese translation are ready."


theme = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    body_background_fill="#f7f3ef",
    block_background_fill="rgba(255, 252, 248, 0.96)",
    block_border_color="#ddd2c7",
    block_radius="24px",
    button_primary_background_fill="#c9785d",
    button_primary_background_fill_hover="#b9634a",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e6ddd5",
    button_secondary_text_color="#4e413b",
    input_background_fill="#fffaf6",
    input_border_color="#d8c8bb",
)


with gr.Blocks(theme=theme, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <div class="hero-card">
              <div class="hero-kicker">Curated Discovery</div>
              <div class="hero-title">Find Something You'd Want To Buy</div>
              <p class="hero-subtitle">
                Search by text or image, skim a quick shopping summary, and only generate a detailed answer when you want to dig deeper.
              </p>
            </div>
            """
        )
        status_box = gr.Textbox(label="Status", value="Starting runtime warmup...", interactive=False, elem_id="status-box")
        cached_results = gr.State([])
        with gr.Row():
            text_input = gr.Textbox(label="Text Query", placeholder="Search for a product / 输入中文商品描述")
            image_input = gr.Image(label="Query Image", type="filepath")
            top_k_input = gr.Slider(label="Top-K", minimum=1, maximum=20, value=10, step=1)
        gr.Markdown("<div class='section-hint'>输入风格、颜色、材质或上传商品图，先快速看结果，再按需生成详细回答。</div>")
        with gr.Row():
            search_button = gr.Button("Search", elem_id="search-btn")
            detail_button = gr.Button("Generate Detailed Answer", elem_id="detail-btn")
        product_cards = gr.HTML(label="Product Cards", elem_id="product-cards")
        table = gr.Dataframe(headers=["product_id", "title", "brand", "product_type", "score"], label="Top-K Results", interactive=False)
        quick_summary_box = gr.Textbox(label="Quick Summary", lines=6)
        answer_box = gr.Textbox(label="Detailed Answer", lines=8)
        search_button.click(
            fn=run_search,
            inputs=[text_input, image_input, top_k_input],
            outputs=[product_cards, table, quick_summary_box, answer_box, cached_results],
        )
        detail_button.click(
            fn=run_detailed_answer,
            inputs=[text_input, cached_results],
            outputs=[answer_box],
        )
        demo.load(fn=warmup_runtime, inputs=None, outputs=status_box)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
