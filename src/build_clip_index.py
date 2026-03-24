import os
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from common import build_faiss_index, ensure_directories, l2_normalize, save_faiss_index, save_pickle
from config import CLIP_IMAGE_INDEX, CLIP_IMAGE_META, CLIP_MODEL_NAME, CLIP_TEXT_INDEX, CLIP_TEXT_META, PRODUCTS_CLEAN_PARQUET


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 8


def batched(items: List, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def encode_images(model: CLIPModel, processor: CLIPProcessor, image_paths: List[str]) -> np.ndarray:
    vectors = []
    for batch_paths in tqdm(list(batched(image_paths, BATCH_SIZE)), desc="Encoding CLIP images"):
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            embeds = model.visual_projection(pooled)
        vectors.append(embeds.detach().cpu().numpy())
    return l2_normalize(np.concatenate(vectors, axis=0))


def encode_texts(model: CLIPModel, processor: CLIPProcessor, texts: List[str]) -> np.ndarray:
    vectors = []
    for batch_texts in tqdm(list(batched(texts, BATCH_SIZE)), desc="Encoding CLIP texts"):
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        with torch.no_grad():
            text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = text_outputs.pooler_output
            embeds = model.text_projection(pooled)
        vectors.append(embeds.detach().cpu().numpy())
    return l2_normalize(np.concatenate(vectors, axis=0))


def main() -> None:
    ensure_directories()
    dataframe = pd.read_parquet(PRODUCTS_CLEAN_PARQUET)
    dataframe = dataframe[dataframe["image_path"].apply(os.path.exists)].reset_index(drop=True)

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    image_vectors = encode_images(model, processor, dataframe["image_path"].tolist())
    image_index = build_faiss_index(image_vectors)
    save_faiss_index(image_index, CLIP_IMAGE_INDEX)
    meta_records = dataframe[["product_id", "image_path", "image_url", "title", "brand", "product_type", "color", "material", "style", "description", "text_input"]].to_dict(orient="records")
    save_pickle(CLIP_IMAGE_META, meta_records)

    text_vectors = encode_texts(model, processor, dataframe["text_input"].tolist())
    text_index = build_faiss_index(text_vectors)
    save_faiss_index(text_index, CLIP_TEXT_INDEX)
    save_pickle(CLIP_TEXT_META, meta_records)

    print(CLIP_IMAGE_INDEX)
    print(CLIP_IMAGE_META)
    print(CLIP_TEXT_INDEX)
    print(CLIP_TEXT_META)


if __name__ == "__main__":
    main()
