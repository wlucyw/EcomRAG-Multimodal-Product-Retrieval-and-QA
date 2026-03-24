PROJECT_ROOT = r"D:\ecommerce-multimodal-rag"
DATA_ROOT = rf"{PROJECT_ROOT}\data"
RAW_ROOT = rf"{DATA_ROOT}\raw\abo"
PROCESSED_ROOT = rf"{DATA_ROOT}\processed"
IMAGES_ROOT = rf"{RAW_ROOT}\images"
INDEX_ROOT = rf"{PROJECT_ROOT}\indexes"
APP_ROOT = rf"{PROJECT_ROOT}\app"
OUTPUT_ROOT = rf"{PROJECT_ROOT}\outputs"

ABO_METADATA_URLS = [
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_0.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_1.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_2.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_3.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_4.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_5.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_6.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_7.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_8.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_9.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_10.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_11.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_12.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_13.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_14.json.gz",
    "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_15.json.gz",
]

ABO_IMAGES_METADATA_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz"

PRODUCTS_SUBSET_CSV = rf"{PROCESSED_ROOT}\products_subset.csv"
PRODUCTS_CLEAN_PARQUET = rf"{PROCESSED_ROOT}\products_clean.parquet"
ID2META_JSON = rf"{PROCESSED_ROOT}\id2meta.json"

CLIP_IMAGE_INDEX = rf"{INDEX_ROOT}\clip_image.index"
CLIP_IMAGE_META = rf"{INDEX_ROOT}\image_meta.pkl"
CLIP_TEXT_INDEX = rf"{INDEX_ROOT}\clip_text.index"
CLIP_TEXT_META = rf"{INDEX_ROOT}\clip_text_meta.pkl"
TEXT_BGE_INDEX = rf"{INDEX_ROOT}\text_bge.index"
TEXT_BGE_META = rf"{INDEX_ROOT}\text_meta.pkl"

LOCAL_LLM_MODEL = "google/flan-t5-small"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
ZH_EN_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-zh-en"
EN_ZH_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-zh"

MAX_PRODUCTS = 10000
IMAGE_DOWNLOAD_TIMEOUT = 30
