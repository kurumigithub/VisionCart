from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.api import serpapi_google_shopping_search


@dataclass(frozen=True)
class StylistOutput:
    """
    Minimal contract for what the stylist agent should output.
    This keeps procurement decoupled from how you compute style/mood/vibe.
    """

    style_profile: str
    aesthetic: List[str]   # mood/vibe descriptors — NOT product names (e.g. "cottagecore", "boho")
    colors: List[str]
    materials: List[str]
    products: List[str]    # purchasable item types — what to search for (e.g. "planters")
    budget_max: Optional[float] = None
    budget_currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def build_queries(style: StylistOutput) -> List[str]:
    """
    Returns one query per product type detected by the stylist.

    Each query = product + random material + random color + random aesthetic.
    Number of queries == len(style.products), so API calls scale with how
    many product types the stylist identified, not a fixed permutation count.
    """
    products = style.products or [""]
    materials = style.materials or [""]
    colors = style.colors or [""]
    aesthetics = [a for a in style.aesthetic if a]

    seen_q: set = set()
    queries: List[str] = []

    for product in products:
        parts = [product, random.choice(materials), random.choice(colors)]
        if aesthetics:
            parts.append(random.choice(aesthetics))
        q = " ".join(p for p in parts if p).strip()
        q = " ".join(q.split())
        if q and q not in seen_q:
            seen_q.add(q)
        queries.append(q)

    return queries


def _trim_product_json(items: List[Dict[str, Any]], *, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "image_url": it.get("image_url", ""),
                "product_name": it.get("product_name", ""),
                "price": it.get("price"),
                "link": it.get("product_url", ""),
                "tags": it.get("tags") or [],
            }
        )
        if len(out) >= n:
            break
    return out


def _resolve_style(state: Dict[str, Any]) -> StylistOutput:
    """
    Resolve StylistOutput from state.
    Expects stylist_output dict in state (from real or mock stylist agent).
    """
    stylist_dict = state.get("stylist_output")
    if not stylist_dict:
        raise ValueError("stylist_output is required in state")
    return StylistOutput(
        style_profile=stylist_dict.get("style_profile", ""),
        aesthetic=list(stylist_dict.get("aesthetic") or []),
        colors=list(stylist_dict.get("colors") or []),
        materials=list(stylist_dict.get("materials") or []),
        products=list(stylist_dict.get("products") or []),
        budget_max=stylist_dict.get("budget_max"),
        budget_currency=stylist_dict.get("budget_currency"),
    )


def run(state: Dict[str, Any]) -> str:
    """
    Procurement agent: read stylist output from state, query SerpAPI,
    and return product candidates as a JSON string.

    Reads from state:
      - stylist_output (dict): output from real or mock stylist agent
      - num_products (int, optional): number of items to return (default 10)

    Query count equals the number of product types detected by the stylist.
    Results are merged via round-robin across query buckets so every product
    type contributes evenly to the final list.

    Returns a JSON string with keys:
      - procurement_queries (list)
      - procurement_products (list): trimmed list with image_url, product_name, price, link, tags
      - style_profile (str)
    """
    if not os.environ.get("SERPAPI_API_KEY"):
        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]

            load_dotenv()
        except Exception:
            pass

    api_key = os.environ.get("SERPAPI_API_KEY")
    num_products = int(state.get("num_products") or 10)

    style = _resolve_style(state)
    queries = build_queries(style)

    # Fetch results per query into separate buckets, deduplicating globally.
    seen_products: set = set()
    per_query_buckets: List[List[Dict[str, Any]]] = []

    for q in queries:
        bucket: List[Dict[str, Any]] = []
        for it in serpapi_google_shopping_search(
            api_key=api_key,
            query=q,
            num=max(num_products, 10),
        ):
            key = (
                (it.get("product_url") or "").strip(),
                (it.get("product_name") or "").strip().lower(),
            )
            if key not in seen_products:
                seen_products.add(key)
                bucket.append(it)
        per_query_buckets.append(bucket)

    # Round-robin across buckets so every product type contributes evenly.
    merged: List[Dict[str, Any]] = []
    iterators = [iter(b) for b in per_query_buckets]
    while len(merged) < num_products:
        advanced = False
        for it in iterators:
            if len(merged) >= num_products:
                break
            item = next(it, None)
            if item is not None:
                merged.append(item)
                advanced = True
        if not advanced:
            break

    trimmed = _trim_product_json(merged, n=num_products)
    result = {
        "procurement_queries": queries,
        "procurement_products": trimmed,
        "style_profile": style.style_profile,
    }
    return json.dumps(result, indent=2)
