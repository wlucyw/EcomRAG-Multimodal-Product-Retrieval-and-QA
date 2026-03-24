import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from transformers.utils import logging as hf_logging

from common import load_faiss_index, load_pickle, load_pil_image, l2_normalize, search_index
from config import CLIP_IMAGE_INDEX, CLIP_IMAGE_META, CLIP_MODEL_NAME, TEXT_BGE_INDEX, TEXT_BGE_META, TEXT_MODEL_NAME
from query_utils import normalize_query


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_processor = None
_text_model = None
_text_tokenizer = None
_image_index = None
_image_meta = None
_text_index = None
_text_meta = None


def get_clip_components():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor


def get_text_components():
    global _text_model, _text_tokenizer
    if _text_model is None or _text_tokenizer is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
                _text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
        _text_model.eval()
    return _text_model, _text_tokenizer


def get_image_index():
    global _image_index, _image_meta
    if _image_index is None or _image_meta is None:
        _image_index = load_faiss_index(CLIP_IMAGE_INDEX)
        _image_meta = load_pickle(CLIP_IMAGE_META)
    return _image_index, _image_meta


def get_text_index():
    global _text_index, _text_meta
    if _text_index is None or _text_meta is None:
        _text_index = load_faiss_index(TEXT_BGE_INDEX)
        _text_meta = load_pickle(TEXT_BGE_META)
    return _text_index, _text_meta


def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_text_query(query: str):
    query = normalize_query(query)
    model, tokenizer = get_text_components()
    inputs = tokenizer(
        [f"Represent this product for retrieval: {query}"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        vectors = _pool(outputs.last_hidden_state, inputs["attention_mask"]).detach().cpu().numpy()
    return l2_normalize(vectors)


def encode_clip_text_query(query: str):
    query = normalize_query(query)
    model, processor = get_clip_components()
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vectors = model.text_projection(text_outputs.pooler_output).detach().cpu().numpy()
    return l2_normalize(vectors)


def encode_clip_image_query(image_path: str):
    model, processor = get_clip_components()
    image = load_pil_image(image_path)
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        vectors = model.visual_projection(vision_outputs.pooler_output).detach().cpu().numpy()
    return l2_normalize(vectors)


def pack_results(scores: List[float], indices: List[int], meta: List[Dict], top_k: int) -> List[Dict]:
    results: List[Dict] = []
    for score, index in zip(scores, indices):
        if index < 0 or index >= len(meta):
            continue
        item = dict(meta[index])
        item["score"] = float(score)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def text_to_image_search(query, top_k=10):
    image_index, image_meta = get_image_index()
    query_vector = encode_clip_text_query(query)
    scores, indices = search_index(image_index, query_vector, top_k)
    return pack_results(scores, indices, image_meta, top_k)


def image_to_image_search(image_path, top_k=10):
    image_index, image_meta = get_image_index()
    query_vector = encode_clip_image_query(image_path)
    scores, indices = search_index(image_index, query_vector, top_k)
    return pack_results(scores, indices, image_meta, top_k)


def text_rag_search(query, top_k=10):
    text_index, text_meta = get_text_index()
    query_vector = encode_text_query(query)
    scores, indices = search_index(text_index, query_vector, top_k)
    return pack_results(scores, indices, text_meta, top_k)


def warmup_retrieval_models() -> None:
    get_text_components()
    get_clip_components()
    get_text_index()
    get_image_index()
