import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

import faiss
import numpy as np
from PIL import Image

from config import APP_ROOT, DATA_ROOT, IMAGES_ROOT, INDEX_ROOT, OUTPUT_ROOT, PROCESSED_ROOT, PROJECT_ROOT, RAW_ROOT


def ensure_directories() -> None:
    for path in [
        PROJECT_ROOT,
        DATA_ROOT,
        RAW_ROOT,
        IMAGES_ROOT,
        PROCESSED_ROOT,
        INDEX_ROOT,
        APP_ROOT,
        OUTPUT_ROOT,
    ]:
        os.makedirs(path, exist_ok=True)


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_text_value(field) -> str:
    if field is None:
        return ""
    if isinstance(field, str):
        return normalize_text(field)
    if isinstance(field, dict):
        for key in ["value", "name", "display_value"]:
            if key in field and field[key] is not None:
                return normalize_text(field[key])
        return ""
    if isinstance(field, list):
        values: List[str] = []
        for item in field:
            if isinstance(item, dict):
                raw = item.get("value")
                if raw is None and item.get("standardized_values"):
                    raw = ", ".join([str(v) for v in item["standardized_values"] if v])
                if raw is not None:
                    values.append(normalize_text(raw))
            elif item is not None:
                values.append(normalize_text(item))
        return normalize_text(" | ".join([v for v in values if v]))
    return normalize_text(field)


def download_file(url: str, output_path: str, timeout: int = 60, skip_if_exists: bool = True) -> str:
    ensure_directories()
    if skip_if_exists and os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response, open(output_path, "wb") as file_obj:
        file_obj.write(response.read())
    return output_path


def try_download_image(url: str, output_path: str, timeout: int = 30) -> bool:
    if os.path.exists(output_path):
        return True
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout) as response, open(output_path, "wb") as file_obj:
            file_obj.write(response.read())
        return True
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def save_pickle(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_obj:
        pickle.dump(payload, file_obj)


def load_pickle(path: str):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index


def save_faiss_index(index: faiss.Index, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def search_index(index: faiss.Index, query_vector: np.ndarray, top_k: int):
    scores, indices = index.search(query_vector.astype(np.float32), top_k)
    return scores[0].tolist(), indices[0].tolist()


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def safe_basename(path: str) -> str:
    return Path(path).name


def call_subprocess_python(script_path: str, args: List[str]) -> str:
    command = [sys.executable, script_path] + args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, check=True, capture_output=True, text=False, env=env)
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    if stdout:
        return stdout
    return completed.stderr.decode("utf-8", errors="replace").strip()
