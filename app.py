"""
VisionCart — Chainlit chat interface.

Flow:
  1. User types a product type (bag, shirt, purse, …)
  2. A random aesthetic style is assigned
  3. Procurement: LLM generates search queries → live SerpAPI Google Shopping results
  4. Ranker/critic scores and filters candidates
  5. If critic fires feedback → second procurement pass against SerpAPI
  6. Output agent writes the shopping summary
  7. Product cards are rendered; loop resets for the next query

No stylist agent is used — style profiles are loaded from STYLE_DEFINITIONS.
Requires SERPAPI_API_KEY and (OLLAMA_MODEL or HF_TOKEN) in .env or environment.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

# Load .env early so SERPAPI_API_KEY / HF_TOKEN / OLLAMA_MODEL are available
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from test_eval import (  # noqa: E402
    STYLE_DEFINITIONS,
    make_style_profile,
    make_stylist_output,
)

STYLE_DISPLAY: Dict[str, str] = {
    "dark_academia":         "Dark Academia",
    "quiet_luxury":          "Quiet Luxury",
    "y2k_streetwear":        "Y2K Streetwear",
    "coastal_mediterranean": "Coastal Mediterranean",
    "maximalist_eclectic":   "Maximalist Eclectic",
    "dark_moody_organic":    "Dark Moody Organic",
    "mid_century_modern":    "Mid Century Modern",
    "vintage_bohemian":      "Vintage Bohemian",
    "japandi":               "Japandi",
    "cottagecore":           "Cottagecore",
}
STYLE_LABELS = list(STYLE_DISPLAY.keys())


# ── Pipeline stages (synchronous — run in executor) ───────────────────────────

def _stage_generate_queries(user_query: str, style_label: str) -> Tuple[Dict[str, Any], List[str]]:
    """Ask the LLM to generate Google Shopping search queries for this style + user query."""
    from agents.procurement import StylistOutput, build_queries

    stylist_dict = make_stylist_output(style_label)
    stylist_dict["style_profile"] = (
        f"{stylist_dict['style_profile']} "
        f"The user is specifically looking for: {user_query}. "
        f"Make sure all queries target that product."
    )
    style = StylistOutput(
        style_profile=stylist_dict["style_profile"],
        aesthetic=stylist_dict["aesthetic"],
        colors=stylist_dict["colors"],
        materials=stylist_dict["materials"],
    )
    queries = build_queries(style, num_queries=3)
    return stylist_dict, queries


def _stage_serpapi_search(
    queries: List[str], num_results: int = 15, id_offset: int = 0
) -> List[Dict[str, Any]]:
    """Hit SerpAPI Google Shopping for each query and return a deduplicated product list.

    Each product gets a stable product_id so ranker_critic._match_originals can
    look them back up and preserve image_url / price / link in the final output.
    """
    from tools.api import serpapi_google_shopping_search

    api_key = os.environ.get("SERPAPI_API_KEY")
    seen: set = set()
    products: List[Dict[str, Any]] = []

    for q in queries:
        for item in serpapi_google_shopping_search(api_key=api_key, query=q, num=num_results):
            key = (
                (item.get("product_url") or "").strip(),
                (item.get("product_name") or "").strip().lower(),
            )
            if key not in seen:
                seen.add(key)
                products.append({
                    "product_id":   f"p_{id_offset + len(products)}",
                    "product_name": item.get("product_name", ""),
                    "image_url":    item.get("image_url", ""),
                    "price":        item.get("price"),
                    "link":         item.get("product_url", ""),
                    "tags":         item.get("tags") or [],
                })

    return products


def _stage_rank(style_label: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
    from agents.ranker_critic import run as ranker_run

    return ranker_run({
        "style_profile":      make_style_profile(style_label),
        "candidate_products": products,
    })


def _stage_feedback_pass(
    stylist_dict: Dict[str, Any],
    feedback: str,
    existing_products: List[Dict[str, Any]],
    style_label: str,
    num_results: int = 15,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Re-run query generation with critic feedback, hit SerpAPI again, re-rank combined pool."""
    from agents.procurement import StylistOutput, build_queries
    from agents.ranker_critic import run as ranker_run

    style = StylistOutput(
        style_profile=stylist_dict["style_profile"],
        aesthetic=stylist_dict["aesthetic"],
        colors=stylist_dict["colors"],
        materials=stylist_dict["materials"],
    )
    pass2_queries = build_queries(style, num_queries=3, critic_feedback=feedback)

    new_products = _stage_serpapi_search(
        pass2_queries, num_results=num_results, id_offset=len(existing_products)
    )

    # Merge with existing, dedup by (url, name)
    seen = {
        ((p.get("link") or "").strip(), (p.get("product_name") or "").strip().lower())
        for p in existing_products
    }
    combined = list(existing_products)
    for p in new_products:
        key = ((p.get("link") or "").strip(), (p.get("product_name") or "").strip().lower())
        if key not in seen:
            seen.add(key)
            combined.append(p)

    result = ranker_run({
        "style_profile":      make_style_profile(style_label),
        "candidate_products": combined,
    })
    return pass2_queries, combined, result


def _stage_output(style_label: str, ranked_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    from agents.output import run as output_run

    defn = STYLE_DEFINITIONS.get(style_label, {})
    return output_run({
        "style_profile":   defn.get("style_summary", style_label),
        "ranked_products": ranked_products,
    })


# ── Status helper ─────────────────────────────────────────────────────────────

async def _status(text: str) -> None:
    await cl.Message(content=text).send()


# ── Main pipeline orchestrator ────────────────────────────────────────────────

async def _run_pipeline(user_query: str, style_label: str) -> None:
    cl.user_session.set("state", "running")
    display = STYLE_DISPLAY.get(style_label, style_label)
    loop = asyncio.get_running_loop()

    try:
        await _status(f"**Style assigned:** {display} | **Looking for:** {user_query}")

        # ── Stage 1: LLM query generation ────────────────────────────────────
        await _status("🔍 Generating search queries with LLM…")
        stylist_dict, queries = await loop.run_in_executor(
            None, _stage_generate_queries, user_query, style_label
        )
        await _status("**Queries:**\n" + "\n".join(f"• `{q}`" for q in queries))

        # ── Stage 2: SerpAPI → Google Shopping ───────────────────────────────
        await _status("🛍️ Hitting SerpAPI — fetching live Google Shopping results…")
        products = await loop.run_in_executor(None, _stage_serpapi_search, queries)
        await _status(f"✓ **{len(products)}** products retrieved from Google Shopping")

        if not products:
            await _status("No products returned from SerpAPI. Check your SERPAPI_API_KEY.")
            cl.user_session.set("state", "awaiting_product")
            return

        # ── Stage 3: Rank + critique ──────────────────────────────────────────
        await _status("⚖️ Ranking candidates and running critic…")
        rank_result = await loop.run_in_executor(None, _stage_rank, style_label, products)
        ranked: List[Dict[str, Any]] = rank_result.get("ranked_products", [])
        feedback: Optional[str] = rank_result.get("critic_feedback")
        n_rejected = len(rank_result.get("rejected_products", []))
        await _status(
            f"✓ Ranker done — **accepted:** {len(ranked)} | **rejected:** {n_rejected}"
        )

        # ── Stage 4: Feedback loop ────────────────────────────────────────────
        if feedback:
            await _status(f"🔄 Critic fired — running second procurement pass\n> {feedback}")
            pass2_queries, combined, rank_result2 = await loop.run_in_executor(
                None,
                _stage_feedback_pass,
                stylist_dict, feedback, products, style_label,
            )
            ranked = rank_result2.get("ranked_products", [])
            n_rej2 = len(rank_result2.get("rejected_products", []))
            await _status(
                "**Pass 2 queries:**\n"
                + "\n".join(f"• `{q}`" for q in pass2_queries)
                + f"\n✓ Pass 2 — **accepted:** {len(ranked)} | **rejected:** {n_rej2}"
            )
        else:
            await _status("✓ No critic feedback — results look good")

        # ── Stage 5: Output agent ─────────────────────────────────────────────
        await _status("✍️ Generating shopping summary…")
        out = await loop.run_in_executor(None, _stage_output, style_label, ranked)

        # ── Render results ────────────────────────────────────────────────────
        summary_text = out.get("output_text", "")
        if summary_text:
            await cl.Message(content=summary_text).send()

        for p in out.get("output_products", []):
            lines: List[str] = [f"### {p['name']}"]
            if p.get("price"):
                lines.append(f"**Price:** {p['price']}")
            if p.get("tags"):
                lines.append(f"**Tags:** {', '.join(str(t) for t in p['tags'][:5])}")
            if p.get("score") is not None:
                lines.append(f"**Match score:** {p['score']}")
            if p.get("url"):
                lines.append(f"[View product]({p['url']})")

            elements: List[cl.Element] = []
            if p.get("image_url"):
                elements.append(cl.Image(url=p["image_url"], name=p["name"], display="inline"))

            await cl.Message(content="\n".join(lines), elements=elements).send()

        cl.user_session.set("state", "awaiting_product")
        await cl.Message(content="---\nWhat else are you looking for?").send()

    except Exception as exc:
        await cl.Message(
            content=f"**Pipeline error** — `{type(exc).__name__}: {exc}`"
        ).send()
        cl.user_session.set("state", "awaiting_product")


# ── Chainlit event handlers ───────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("state", "awaiting_product")
    await cl.Message(
        content=(
            "## VisionCart\n\n"
            "What product are you looking for?\n"
            "*(e.g. bag, shirt, shoes, purse, candle, blazer…)*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    state: str = cl.user_session.get("state", "awaiting_product")

    if state == "running":
        await cl.Message(content="Pipeline is still running — hang tight!").send()
        return

    if state == "awaiting_product":
        user_query = message.content.strip()
        if not user_query:
            return
        cl.user_session.set("user_query", user_query)
        cl.user_session.set("state", "awaiting_style")

        style_list = "\n".join(
            f"`{i + 1}` — **{STYLE_DISPLAY[label]}**"
            for i, label in enumerate(STYLE_LABELS)
        )
        await cl.Message(
            content=f"Got it — **\"{user_query}\"**. Pick a style:\n\n{style_list}"
        ).send()

    elif state == "awaiting_style":
        raw = message.content.strip()

        # Match by number (1-based)
        matched: Optional[str] = None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(STYLE_LABELS):
                matched = STYLE_LABELS[idx]

        # Fallback: match by name
        if not matched:
            norm = raw.lower().replace(" ", "_").replace("-", "_")
            matched = next(
                (l for l in STYLE_LABELS if norm == l or norm in l or l in norm), None
            )

        if not matched:
            style_list = "\n".join(
                f"`{i + 1}` — **{STYLE_DISPLAY[l]}**"
                for i, l in enumerate(STYLE_LABELS)
            )
            await cl.Message(
                content=f"Type a number 1–{len(STYLE_LABELS)}:\n\n{style_list}"
            ).send()
            return

        user_query = cl.user_session.get("user_query", "")
        await _run_pipeline(user_query, matched)
