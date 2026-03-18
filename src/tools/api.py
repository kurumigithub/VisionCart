from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


Money = Tuple[Optional[float], Optional[str]]

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "on", "with",
    "to", "by", "at", "is", "it", "its", "as", "be", "set", "pack",
    "pcs", "pc", "piece", "pieces", "lot", "new", "sale", "free",
}


def _extract_title_tags(title: str) -> List[str]:
    """Extract descriptive keyword tags from a product title."""
    # Split on common delimiters, lowercase, strip whitespace
    tokens = re.split(r"[\-,|/&+()[\]{}]+", title.lower())
    tags = []
    for token in tokens:
        token = token.strip()
        if token and token not in _STOP_WORDS and not token.isdigit():
            tags.append(token)
    return tags


def _parse_price_to_money(price: Any) -> Money:
    """
    Best-effort parsing for common SerpAPI price strings, e.g. "$19.99", "£12.00".
    Returns (value, currency_code) or (None, None) if unknown.
    """
    if price is None:
        return (None, None)
    if isinstance(price, (int, float)):
        return (float(price), None)
    if not isinstance(price, str):
        return (None, None)

    s = price.strip()
    if not s:
        return (None, None)

    currency_map = {
        "$": "USD",
        "£": "GBP",
        "€": "EUR",
        "¥": "JPY",
        "₹": "INR",
        "₩": "KRW",
        "₫": "VND",
        "₺": "TRY",
        "R$": "BRL",
        "C$": "CAD",
        "A$": "AUD",
    }

    currency = None
    for sym, code in sorted(currency_map.items(), key=lambda x: -len(x[0])):
        if s.startswith(sym):
            currency = code
            s = s[len(sym) :].strip()
            break

    # Extract first numeric token (handles commas)
    m = re.search(r"(\d[\d,]*\.?\d*)", s)
    if not m:
        return (None, currency)
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return (None, currency)
    return (value, currency)


def serpapi_google_shopping_search(
    *,
    api_key: str,
    query: str,
    num: int = 10,
    country: str = "us",
    language: str = "en",
    timeout_s: int = 30,
) -> List[Dict[str, Any]]:
    """
    Calls SerpAPI's Google Shopping engine and returns a *normalized* list of items:
      - image_url
      - product_name
      - price (float or None)
      - currency (str or None)
      - product_url
      - source (merchant, optional)
      - raw (original item)
    """
    if not api_key:
        raise ValueError("Missing SerpAPI api_key")
    if not query or not query.strip():
        return []

    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": api_key,
        "gl": country,
        "hl": language,
    }
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("shopping_results") or []
    out: List[Dict[str, Any]] = []
    for it in items[: max(0, num)]:
        price_val, currency = _parse_price_to_money(it.get("price"))
        tags = _extract_title_tags(it.get("title") or it.get("name") or "")
        out.append(
            {
                "image_url": it.get("thumbnail") or it.get("image"),
                "product_name": it.get("title") or it.get("name") or "",
                "price": price_val,
                "currency": currency,
                "product_url": quote(it.get("link") or it.get("product_link") or "", safe=":/?=&#%+@"),
                "source": it.get("source"),
                "tags": tags,
                "raw": it,
            }
        )

    # Filter out empty rows (no name + no url)
    return [p for p in out if p.get("product_name") or p.get("product_url")]