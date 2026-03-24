import pandas as pd

from common import ensure_directories, save_json
from config import ID2META_JSON, PRODUCTS_CLEAN_PARQUET, PRODUCTS_SUBSET_CSV


TEXT_FIELDS = [
    "title",
    "brand",
    "product_type",
    "color",
    "material",
    "style",
    "description",
]


def build_text_input(row: pd.Series) -> str:
    return " ".join([f"{field}: {str(row[field]).strip()}" for field in TEXT_FIELDS if str(row[field]).strip()])


def main() -> None:
    ensure_directories()
    dataframe = pd.read_csv(PRODUCTS_SUBSET_CSV)
    dataframe = dataframe.dropna(subset=["product_id", "image_url", "image_path", "title", "brand", "product_type", "color", "material", "style", "description"])
    dataframe = dataframe[dataframe["image_path"].astype(str).str.len() > 0].copy()
    dataframe["text_input"] = dataframe.apply(build_text_input, axis=1)
    dataframe = dataframe[dataframe["text_input"].astype(str).str.len() > 0].reset_index(drop=True)
    dataframe.to_parquet(PRODUCTS_CLEAN_PARQUET, index=False)

    id2meta = {
        str(row["product_id"]): {
            "product_id": str(row["product_id"]),
            "image_url": str(row["image_url"]),
            "image_path": str(row["image_path"]),
            "title": str(row["title"]),
            "brand": str(row["brand"]),
            "product_type": str(row["product_type"]),
            "color": str(row["color"]),
            "material": str(row["material"]),
            "style": str(row["style"]),
            "description": str(row["description"]),
            "text_input": str(row["text_input"]),
        }
        for _, row in dataframe.iterrows()
    }
    save_json(ID2META_JSON, id2meta)
    print(PRODUCTS_CLEAN_PARQUET)
    print(ID2META_JSON)


if __name__ == "__main__":
    main()
