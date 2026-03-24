from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import build_faiss_index, ensure_directories, l2_normalize, save_faiss_index, save_pickle
from config import PRODUCTS_CLEAN_PARQUET, TEXT_BGE_INDEX, TEXT_BGE_META, TEXT_MODEL_NAME


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 if DEVICE == "cuda" else 16


def batched(items: List[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(model: AutoModel, tokenizer: AutoTokenizer, texts: List[str]) -> np.ndarray:
    vectors = []
    for batch_texts in tqdm(list(batched(texts, BATCH_SIZE)), desc="Encoding BGE texts"):
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        vectors.append(embeddings.detach().cpu().numpy())
    return l2_normalize(np.concatenate(vectors, axis=0))


def main() -> None:
    ensure_directories()
    dataframe = pd.read_parquet(PRODUCTS_CLEAN_PARQUET).reset_index(drop=True)
    texts = [f"Represent this product for retrieval: {text}" for text in dataframe["text_input"].tolist()]

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
    model.eval()

    vectors = encode_texts(model, tokenizer, texts)
    index = build_faiss_index(vectors)
    save_faiss_index(index, TEXT_BGE_INDEX)
    save_pickle(
        TEXT_BGE_META,
        dataframe[["product_id", "image_path", "image_url", "title", "brand", "product_type", "color", "material", "style", "description", "text_input"]].to_dict(orient="records"),
    )
    print(TEXT_BGE_INDEX)
    print(TEXT_BGE_META)


if __name__ == "__main__":
    main()
