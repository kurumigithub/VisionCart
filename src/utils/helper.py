"""Shared utility functions used across VisionCart agents."""

from __future__ import annotations

import re
from typing import List


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_product_text(product: dict) -> str:
    """
    Combine product name and tags into a single normalized text blob.

    Checks both 'product_name' (procurement output shape) and 'name'
    (ranker output shape) so it works at every stage of the pipeline.
    """
    name = product.get("product_name") or product.get("name") or ""
    tags = product.get("tags") or []
    return normalize_text(f"{name} {' '.join(tags)}")


def keyword_overlap_score(style_keywords: List[str], product_text: str) -> float:
    """
    Fraction of style_keywords that appear in product_text.

    Both sides are normalized before matching so case and punctuation
    differences are ignored. Returns a float in [0.0, 1.0].
    Returns 0.0 when style_keywords is empty.
    """
    if not style_keywords:
        return 0.0
    normalized = normalize_text(product_text)
    hits = sum(1 for kw in style_keywords if normalize_text(kw) in normalized)
    return min(1.0, hits / len(style_keywords))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two float vectors.

    Returns a float in [-1.0, 1.0]. Returns 0.0 when either vector is
    zero, the lengths differ, or numpy is unavailable.
    numpy is lazy-imported so this file has no hard dependencies.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import numpy as np
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        return 0.0
