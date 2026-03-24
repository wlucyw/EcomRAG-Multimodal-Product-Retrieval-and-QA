from typing import Dict, List

from local_llm import generate_local_answer, is_bad_answer
from query_utils import contains_chinese


def build_context(retrieved_items: List[Dict]) -> str:
    chunks = []
    for index, item in enumerate(retrieved_items, start=1):
        chunks.append(
            "\n".join(
                [
                    f"[Item {index}]",
                    f"product_id: {item.get('product_id', '')}",
                    f"title: {item.get('title', '')}",
                    f"brand: {item.get('brand', '')}",
                    f"product_type: {item.get('product_type', '')}",
                    f"color: {item.get('color', '')}",
                    f"material: {item.get('material', '')}",
                    f"style: {item.get('style', '')}",
                    f"description: {item.get('description', '')}",
                    f"score: {item.get('score', 0.0)}",
                ]
            )
        )
    return "\n\n".join(chunks)


def generate_quick_summary(query, retrieved_items):
    if not retrieved_items:
        return "未找到相关商品。" if contains_chinese(query) else "No relevant products were found."

    top_items = retrieved_items[:3]
    if contains_chinese(query):
        lines = ["根据检索结果，最相关的商品包括："]
        for item in top_items:
            lines.append(
                f"{item.get('title', '')}，品牌 {item.get('brand', '')}，类别 {item.get('product_type', '')}。"
            )
        return "\n".join(lines)

    lines = ["The most relevant retrieved products are:"]
    for item in top_items:
        lines.append(
            f"{item.get('title', '')}, brand {item.get('brand', '')}, type {item.get('product_type', '')}."
        )
    return "\n".join(lines)


def generate_answer(query, retrieved_items):
    if not retrieved_items:
        return "未找到相关商品。" if contains_chinese(query) else "No relevant products were found."

    if contains_chinese(query):
        return generate_quick_summary(query, retrieved_items)

    context = build_context(retrieved_items[:3])
    answer = generate_local_answer(query, context)
    if not is_bad_answer(answer):
        return answer
    return generate_quick_summary(query, retrieved_items)
