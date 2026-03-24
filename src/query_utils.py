import os
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from io import StringIO

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from config import EN_ZH_TRANSLATION_MODEL, ZH_EN_TRANSLATION_MODEL


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_zh_en_tokenizer = None
_zh_en_model = None


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def get_zh_en_translator():
    global _zh_en_model, _zh_en_tokenizer
    if _zh_en_model is None or _zh_en_tokenizer is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _zh_en_tokenizer = AutoTokenizer.from_pretrained(ZH_EN_TRANSLATION_MODEL)
                _zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(ZH_EN_TRANSLATION_MODEL).to(DEVICE)
        _zh_en_model.eval()
    return _zh_en_model, _zh_en_tokenizer


@lru_cache(maxsize=256)
def translate_zh_to_en(text: str) -> str:
    if not contains_chinese(text):
        return text
    model, tokenizer = get_zh_en_translator()
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4, do_sample=False)
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return translated or text



@lru_cache(maxsize=256)
def normalize_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return ""
    return translate_zh_to_en(query) if contains_chinese(query) else query


def answer_language_hint(query: str) -> str:
    return "Answer in Simplified Chinese." if contains_chinese(query) else "Answer in English."


def warmup_translation() -> None:
    get_zh_en_translator()
