from __future__ import annotations

import json
import os
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
    keywords: List[str]
    colors: List[str]
    materials: List[str]
    categories: List[str]
    budget_max: Optional[float] = None
    budget_currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def build_queries(style: StylistOutput) -> List[str]:
    base_terms = []
    base_terms.extend(style.categories[:2])
    base_terms.extend(style.materials[:1])
    base_terms.extend(style.colors[:1])
    base_terms.extend(style.keywords[:2])

    base = " ".join(t for t in base_terms if t).strip()
    base = base or style.style_profile

    # Use remaining keywords as dynamic suffixes instead of hardcoded terms.
    suffixes = [kw for kw in style.keywords[2:] if kw]
    queries = [base] + [f"{base} {suffix}" for suffix in suffixes]

    seen = set()
    out: List[str] = []
    for q in queries:
        q2 = " ".join(q.split())
        if q2 and q2 not in seen:
            seen.add(q2)
            out.append(q2)
    return out


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
        keywords=list(stylist_dict.get("keywords") or []),
        colors=list(stylist_dict.get("colors") or []),
        materials=list(stylist_dict.get("materials") or []),
        categories=list(stylist_dict.get("categories") or []),
        budget_max=stylist_dict.get("budget_max"),
        budget_currency=stylist_dict.get("budget_currency"),
    )


def run(state: Dict[str, Any]) -> str:
    """
    Procurement agent: read stylist output from state, query SerpAPI,
    and return product candidates as a JSON string.

    Reads from state:
      - stylist_output (dict, optional): output from real stylist agent
      - num_products (int, optional): number of items to return (default 10)

    Returns a JSON string with keys:
      - procurement_queries (list)
      - procurement_products (list): trimmed list with image_url, product_name, price, link
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

    normalized: List[Dict[str, Any]] = []
    for q in queries:
        normalized.extend(
            serpapi_google_shopping_search(
                api_key=api_key,
                query=q,
                num=max(num_products, 10),
            )
        )

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in normalized:
        key = (
            (it.get("product_url") or "").strip(),
            (it.get("product_name") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    trimmed = _trim_product_json(deduped, n=num_products)
    result = {
        "procurement_queries": queries,
        "procurement_products": trimmed,
        "style_profile": style.style_profile,
    }
    return json.dumps(result, indent=2)
