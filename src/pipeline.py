from typing import Dict, List, Optional

from retrieve import encode_clip_text_query, encode_text_query, get_image_index, get_text_index, pack_results, search_index


def _merge_results(result_groups: List[List[Dict]], weights: List[float], top_k: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for group, weight in zip(result_groups, weights):
        for rank, item in enumerate(group):
            product_id = item["product_id"]
            weighted_score = float(item["score"]) * weight
            if product_id not in merged:
                merged[product_id] = dict(item)
                merged[product_id]["score"] = weighted_score
                merged[product_id]["sources"] = [f"rank_{rank + 1}"]
            else:
                merged[product_id]["score"] += weighted_score
                merged[product_id]["sources"].append(f"rank_{rank + 1}")
    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def search(query_text: Optional[str] = None, query_image: Optional[str] = None, top_k: int = 10):
    result_groups: List[List[Dict]] = []
    weights: List[float] = []

    if query_text:
        text_index, text_meta = get_text_index()
        text_vector = encode_text_query(query_text)
        text_scores, text_indices = search_index(text_index, text_vector, top_k)
        result_groups.append(pack_results(text_scores, text_indices, text_meta, top_k))
        weights.append(0.6 if query_image else 0.5)

        clip_index, clip_meta = get_image_index()
        clip_text_vector = encode_clip_text_query(query_text)
        clip_scores, clip_indices = search_index(clip_index, clip_text_vector, top_k)
        result_groups.append(pack_results(clip_scores, clip_indices, clip_meta, top_k))
        weights.append(0.4 if query_image else 0.5)

    if query_image:
        clip_index, clip_meta = get_image_index()
        from retrieve import encode_clip_image_query

        image_vector = encode_clip_image_query(query_image)
        image_scores, image_indices = search_index(clip_index, image_vector, top_k)
        result_groups.append(pack_results(image_scores, image_indices, clip_meta, top_k))
        weights.append(1.0 if not query_text else 0.5)

    if not result_groups:
        return []
    return _merge_results(result_groups, weights, top_k)
