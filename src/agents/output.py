from __future__ import annotations

import os
from typing import Any, Dict


SYSTEM_PROMPT = """You are a personal shopping assistant for a visual-first retail app.
Your job is to translate AI-ranked product results into a warm,
concise, human-readable summary that explains why the results
match the user's aesthetic and helps them make a decision.

Format your response as:
1. A 1-2 sentence interpretation of the user's style profile
2. Top 3-5 products with a brief explanation of why each fits
3. An optional note if results were sparse or any products were filtered

Keep the tone conversational and specific — reference actual
aesthetic details like color, texture, silhouette, or vibe.
Do not use filler phrases like "Great news!" or "I found some products."
"""


def format_products_for_prompt(ranked_products: list) -> str:
    lines = []
    for i, p in enumerate(ranked_products[:5], 1):
        tags = p.get("tags", [])
        tag_str = ", ".join(tags) if tags else "none"
        lines.append(
            f"{i}. {p.get('name', 'Unknown Product')}\n"
            f"   Score: {p.get('score', 'N/A')}\n"
            f"   Tags: {tag_str}\n"
            f"   Price: {p.get('price', 'N/A')}\n"
            f"   URL: {p.get('url', '')}"
        )
    return "\n\n".join(lines)


def build_output_products(ranked_products: list) -> list:
    results = []
    for p in ranked_products[:5]:
        results.append({
            "name": p.get("name", "Unknown Product"),
            "price": p.get("price", "N/A"),
            "url": p.get("url", ""),
            "image_url": p.get("image_url", ""),
            "score": round(p.get("score"), 3) if p.get("score") is not None else None,
            "tags": p.get("tags", []),
        })
    return results


def _call_ollama(system: str, user: str) -> str:
    import re
    import requests

    model    = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "think":  False,
        "options": {"num_predict": 1024},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["message"].get("content") or resp.json()["message"].get("thinking") or ""
    # Strip any residual <think> blocks from models that ignore think:false
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: generate human-readable output from ranked results via Ollama.

    Reads from state:
        style_profile (str): text description of the user's aesthetic
        ranked_products (list): products sorted by similarity score
        critic_notes (str, optional): any rejections or flags from the critic

    Writes to state:
        output_text (str): narrative summary for the chat interface
        output_products (list): top 3-5 products for UI card rendering
    """
    style_profile  = state.get("style_profile", "")
    ranked_products = state.get("ranked_products", [])
    critic_notes   = state.get("critic_notes", "")

    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    print(f"\n[output] ── run() ── model={model}")
    print(f"[output] ranked_products: {len(ranked_products)}  style_profile: {len(style_profile)} chars")

    if not style_profile:
        return {
            "output_text": (
                "We couldn't determine your style profile. "
                "Try adding more images to your board."
            ),
            "output_products": [],
        }

    if not ranked_products:
        return {
            "output_text": (
                "We couldn't find products that matched your style profile. "
                "Try adding more images to your board."
            ),
            "output_products": [],
        }

    products_block = format_products_for_prompt(ranked_products)
    total = len(ranked_products)
    shown = min(total, 5)
    count_note = (
        f"Showing top {shown} of {total} results"
        if total > shown
        else f"{total} result{'s' if total != 1 else ''} found"
    )

    user_message = (
        f"Style profile: {style_profile}\n\n"
        f"Ranked products ({count_note}):\n{products_block}"
        + (f"\n\nCritic notes: {critic_notes}" if critic_notes else "")
        + "\n\nPlease generate the user-facing summary."
    )

    print(f"\n[output] Sending to Ollama...")
    output_text = _call_ollama(SYSTEM_PROMPT, user_message)

    print(f"\n[output] ── LLM response ─────────────────────────────")
    print(output_text)
    print(f"[output] ─────────────────────────────────────────────")

    output_products = build_output_products(ranked_products)
    return {"output_text": output_text, "output_products": output_products}
