import argparse
import os
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from config import LOCAL_LLM_MODEL
from query_utils import answer_language_hint


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model = None


def get_local_llm():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
                _model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL).to(DEVICE)
        _model.eval()
    return _model, _tokenizer


def generate_local_answer(query: str, context: str) -> str:
    model, tokenizer = get_local_llm()
    prompt = (
        "Answer the ecommerce product question using only the provided context. "
        "If the context is insufficient, say so clearly. "
        f"{answer_language_hint(query)}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer


def is_bad_answer(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    cleaned = re.sub(r"[\s,.;:!?，。！？、]+", "", stripped)
    return not cleaned


def warmup_local_llm() -> None:
    get_local_llm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--context-file", required=True)
    args = parser.parse_args()

    with open(args.context_file, "r", encoding="utf-8") as file_obj:
        context = file_obj.read()

    print(generate_local_answer(args.query, context))


if __name__ == "__main__":
    main()
