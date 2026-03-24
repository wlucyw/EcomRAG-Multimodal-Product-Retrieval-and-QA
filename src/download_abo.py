import csv
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from common import download_file, ensure_directories, extract_text_value, normalize_text, try_download_image
from config import ABO_IMAGES_METADATA_URL, ABO_METADATA_URLS, IMAGE_DOWNLOAD_TIMEOUT, IMAGES_ROOT, MAX_PRODUCTS, PRODUCTS_SUBSET_CSV, RAW_ROOT


LISTINGS_CACHE_DIR = rf"{RAW_ROOT}\metadata"
IMAGES_METADATA_CSV_GZ = rf"{RAW_ROOT}\images_metadata.csv.gz"
DOWNLOAD_WORKERS = 32


def load_image_lookup() -> Dict[str, Dict[str, str]]:
    os.makedirs(LISTINGS_CACHE_DIR, exist_ok=True)
    download_file(ABO_IMAGES_METADATA_URL, IMAGES_METADATA_CSV_GZ, timeout=120)
    lookup: Dict[str, Dict[str, str]] = {}
    with gzip.open(IMAGES_METADATA_CSV_GZ, "rt", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            image_id = row.get("image_id")
            if image_id:
                lookup[image_id] = row
    return lookup


def build_image_url(image_row: Dict[str, str]) -> Optional[str]:
    relative_path = normalize_text(image_row.get("path"))
    if not relative_path:
        return None
    return f"https://amazon-berkeley-objects.s3.amazonaws.com/images/small/{relative_path}"


def build_local_image_path(image_row: Dict[str, str]) -> str:
    relative_path = normalize_text(image_row.get("path"))
    filename = os.path.basename(relative_path)
    return rf"{IMAGES_ROOT}\{filename}"


def extract_attribute_value(item: Dict, attribute_name: str) -> str:
    direct_value = extract_text_value(item.get(attribute_name))
    if direct_value:
        return direct_value

    attributes = item.get("item_keywords", {}) or {}
    if attribute_name in attributes:
        return extract_text_value(attributes.get(attribute_name))

    raw_attributes = item.get("attributes", []) or []
    for attr in raw_attributes:
        name = normalize_text(attr.get("name") or attr.get("attribute_name"))
        if name.lower() == attribute_name.lower():
            return extract_text_value(attr.get("value") or attr.get("values"))
    return ""


def extract_description(item: Dict) -> str:
    for key in ["description", "bullet_point", "item_description"]:
        value = extract_text_value(item.get(key))
        if value:
            return value
    details = item.get("item_details", {}) or {}
    return extract_text_value(details)


def parse_listing_item(item: Dict, image_lookup: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    product_id = normalize_text(item.get("item_id") or item.get("product_id") or item.get("asin"))
    title = normalize_text(item.get("item_name") or item.get("title"))
    brand = extract_text_value(item.get("brand"))
    product_type = extract_text_value(item.get("product_type") or item.get("item_type_name"))
    color = extract_attribute_value(item, "color")
    material = extract_attribute_value(item, "material") or "unknown"
    style = extract_attribute_value(item, "style")
    description = extract_description(item)

    image_ids = item.get("main_image_id") or item.get("other_image_id") or item.get("image_id")
    if isinstance(image_ids, list):
        candidate_ids = [normalize_text(image_id) for image_id in image_ids if image_id]
    else:
        candidate_ids = [normalize_text(image_ids)] if image_ids else []

    image_url = ""
    local_image_path = ""
    for image_id in candidate_ids:
        image_row = image_lookup.get(image_id)
        if image_row:
            image_url = build_image_url(image_row) or ""
            local_image_path = build_local_image_path(image_row)
            break

    required_values = [product_id, image_url, title, brand, product_type, color, material, style, description]
    if not all(required_values):
        return None

    return {
        "product_id": product_id,
        "image_url": image_url,
        "image_path": local_image_path,
        "title": title,
        "brand": brand,
        "product_type": product_type,
        "color": color,
        "material": material,
        "style": style,
        "description": description,
    }


def iter_listing_items(gzip_path: str):
    with gzip.open(gzip_path, "rt", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def stream_listing_records() -> List[Dict]:
    records: List[Dict] = []
    seen_product_ids = set()
    image_lookup = load_image_lookup()
    for url in ABO_METADATA_URLS:
        local_path = rf"{LISTINGS_CACHE_DIR}\{os.path.basename(url)}"
        download_file(url, local_path, timeout=120)
        for item in iter_listing_items(local_path):
            record = parse_listing_item(item, image_lookup)
            if record is None:
                continue
            if record["product_id"] in seen_product_ids:
                continue
            records.append(record)
            seen_product_ids.add(record["product_id"])
            if len(records) >= MAX_PRODUCTS:
                return records
    return records


def _download_one(record: Dict[str, str]) -> Optional[Dict[str, str]]:
    if record["image_path"] and try_download_image(record["image_url"], record["image_path"], timeout=IMAGE_DOWNLOAD_TIMEOUT):
        return record
    return None


def download_images(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    valid_records: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(_download_one, record) for record in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            result = future.result()
            if result is not None:
                valid_records.append(result)
    return valid_records


def main() -> None:
    ensure_directories()
    records = stream_listing_records()
    valid_records = download_images(records)
    dataframe = pd.DataFrame(valid_records)
    dataframe = dataframe.sort_values("product_id").reset_index(drop=True)
    dataframe.to_csv(PRODUCTS_SUBSET_CSV, index=False, encoding="utf-8-sig")
    print(PRODUCTS_SUBSET_CSV)
    print(len(dataframe))


if __name__ == "__main__":
    main()
