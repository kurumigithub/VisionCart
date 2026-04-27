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
    The stylist detects vibes, aesthetics, colors, and materials —
    agent derives what to search for from the style profile via LLM.
    """

    style_profile: str
    aesthetic: List[str]
    colors: List[str]
    materials: List[str]
    budget_max: Optional[float] = None
    budget_currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _build_query_prompt(
    style: StylistOutput,
    num_queries: int,
    critic_feedback: Optional[str],
) -> str:
    budget_line = (
        f"Budget: up to {style.budget_max} {style.budget_currency or 'USD'}"
        if style.budget_max
        else "Budget: no hard limit"
    )
    feedback_section = (
        f"\nThe previous search was evaluated and had these issues — address them in your new queries:\n{critic_feedback}\n"
        if critic_feedback
        else ""
    )
    return f"""You are a search query specialist for an aesthetic shopping platform.

Given a style profile, decide which product categories a shopper would want to buy to achieve this aesthetic, then generate one specific Google Shopping search query per category.

Style profile: {style.style_profile}
Colors: {", ".join(style.colors)}
Materials: {", ".join(style.materials)}
Aesthetics: {", ".join(style.aesthetic)}
{budget_line}
{feedback_section}
Rules:
- Generate exactly {num_queries} queries covering different shoppable product categories that suit this aesthetic.
- Each query must be specific and aesthetic-forward — include material, mood, or style era context.
- Avoid generic terms (e.g. "home decor" alone is too vague).
- Do NOT repeat a query angle from a previous run if critic feedback is provided.

Return a JSON array of strings and nothing else. No explanation, no markdown.
Example: ["handmade terracotta ceramic planter cottagecore", "rattan outdoor lantern boho warm patio"]"""


def _parse_queries(raw: str, num_queries: int) -> List[str]:
    """Parse LLM response into a list of query strings, stripping code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    queries = json.loads(text)
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise RuntimeError(f"LLM returned unexpected format: {raw!r}")
    return [q.strip() for q in queries if q.strip()]


def _build_queries_ollama(
    style: StylistOutput,
    num_queries: int,
    critic_feedback: Optional[str],
) -> List[str]:
    import requests as _requests

    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"[procurement] Calling local ollama (model: {model})...")
    if critic_feedback:
        print(f"[procurement] Retry pass — critic feedback:\n  {critic_feedback}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_query_prompt(style, num_queries, critic_feedback)}],
        "stream": False,
        "options": {"num_predict": 512},
    }
    try:
        resp = _requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")

    raw = resp.json()["message"]["content"]
    print(f"[procurement] Query generation complete. Raw response: {raw}")
    return _parse_queries(raw, num_queries)


def _build_queries_hf(
    style: StylistOutput,
    num_queries: int,
    critic_feedback: Optional[str],
    hf_token: str,
) -> List[str]:
    from huggingface_hub import InferenceClient

    model = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[procurement] Calling HuggingFace Inference API (model: {model})...")
    if critic_feedback:
        print(f"[procurement] Retry pass — critic feedback:\n  {critic_feedback}")

    try:
        client = InferenceClient(model=model, token=hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": _build_query_prompt(style, num_queries, critic_feedback)}],
            max_tokens=512,
        )
    except Exception as e:
        raise RuntimeError(f"HuggingFace Inference API call failed: {e}")

    raw = response.choices[0].message.content
    print(f"[procurement] Query generation complete. Raw response: {raw}")
    return _parse_queries(raw, num_queries)


def build_queries(
    style: StylistOutput,
    num_queries: int,
    critic_feedback: Optional[str] = None,
) -> List[str]:
    """
    Generate search queries from a style profile using an LLM.

    Backend selection (in priority order):
      1. OLLAMA_MODEL env var set  → use local ollama
      2. HF_TOKEN env var set      → use HuggingFace Inference API (Qwen2.5-7B-Instruct)
      3. Neither set               → use local ollama with default model (qwen2.5:7b)

    Override the ollama host with OLLAMA_HOST (default: http://localhost:11434).
    Override the ollama model with OLLAMA_MODEL (default: qwen2.5:7b).
    """
    use_ollama = bool(os.environ.get("OLLAMA_MODEL")) or not os.environ.get("HF_TOKEN")
    if use_ollama:
        return _build_queries_ollama(style, num_queries, critic_feedback)
    return _build_queries_hf(style, num_queries, critic_feedback, os.environ["HF_TOKEN"])


def _shape_products(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip each item to the fields downstream agents need."""
    return [
        {
            "image_url": it.get("image_url", ""),
            "product_name": it.get("product_name", ""),
            "price": it.get("price"),
            "link": it.get("product_url", ""),
            "tags": it.get("tags") or [],
        }
        for it in items
    ]


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
        budget_max=stylist_dict.get("budget_max"),
        budget_currency=stylist_dict.get("budget_currency"),
    )


def run(state: Dict[str, Any]) -> str:
    """
    Procurement agent: read stylist output from state, query SerpAPI,
    and return product candidates as a JSON string.

    On the first pass, the LLM determines which product categories suit the
    aesthetic and generates one query per category. On retry passes (when the
    critic agent routes the pipeline back here), reads critic_feedback from
    state and steers the LLM toward better query angles for failing categories.

    Reads from state:
      - stylist_output (dict): output from stylist agent — style_profile,
        aesthetic, colors, materials, budget_max, budget_currency
      - results_per_query (int, optional): number of results to fetch per query (default 10)
      - num_queries (int, optional): number of search queries / product
        categories to explore (default 5)
      - critic_feedback (str, optional): set by the critic agent on retry;
        describes which categories scored poorly and why

    The full deduplicated candidate pool is passed to the ranker — no trimming
    or round-robin. The ranker and critic decide the final selection.

    Raises RuntimeError if HF_TOKEN is not set or the API call fails.

    Returns a JSON string with keys:
      - procurement_queries (list)
      - procurement_products (list): full candidate pool with image_url, product_name, price, link, tags
      - style_profile (str)
    """
    if not os.environ.get("SERPAPI_API_KEY"):
        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]
            load_dotenv()
        except Exception:
            pass

    api_key = os.environ.get("SERPAPI_API_KEY")
    results_per_query = int(state.get("results_per_query") or 20)
    num_queries = int(state.get("num_queries") or 5)
    critic_feedback: Optional[str] = state.get("critic_feedback")

    style = _resolve_style(state)
    queries = build_queries(style, num_queries=num_queries, critic_feedback=critic_feedback)

    # Fetch all results across queries, deduplicating globally.
    # No trimming or round-robin — pass the full candidate pool to the ranker.
    seen_products: set = set()
    all_products: List[Dict[str, Any]] = []

    print(f"[procurement] Fetching products for {len(queries)} queries (up to {results_per_query} results each)...")
    for i, q in enumerate(queries, 1):
        print(f"[procurement]   [{i}/{len(queries)}] Querying SerpAPI: '{q}'")
        before = len(all_products)
        for it in serpapi_google_shopping_search(
            api_key=api_key,
            query=q,
            num=results_per_query,
        ):
            key = (
                (it.get("product_url") or "").strip(),
                (it.get("product_name") or "").strip().lower(),
            )
            if key not in seen_products:
                seen_products.add(key)
                all_products.append(it)
        print(f"[procurement]          → {len(all_products) - before} new products (pool total: {len(all_products)})")

    print(f"[procurement] Done. Total candidate pool: {len(all_products)} products.")
    result = {
        "procurement_queries": queries,
        "procurement_products": _shape_products(all_products),
        "style_profile": style.style_profile,
    }
    return json.dumps(result, indent=2)
