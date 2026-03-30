import sys
import json
import os
from pathlib import Path

sys.path.insert(0, "src")

from agents.procurement import run

# Convenience: load repo-root .env if present.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Mock stylist outputs — three distinct aesthetics
# ---------------------------------------------------------------------------

STYLE_COTTAGECORE = {
    "style_profile": (
        "Bright spring garden vibe: airy pastels, natural textures, "
        "cottagecore-meets-modern sensibility, light wood accents, and a "
        "relaxed outdoor-living feeling."
    ),
    "aesthetic": ["cottagecore", "boho outdoor", "spring patio", "whimsical garden"],
    "colors": ["sage green", "cream", "terracotta"],
    "materials": ["ceramic", "rattan", "wood"],
    "budget_max": 75.0,
    "budget_currency": "USD",
}

STYLE_JAPANDI = {
    "style_profile": (
        "Calm, stripped-back Japandi interior: warm neutrals, clean lines, "
        "wabi-sabi imperfection, natural materials like linen and pale oak, "
        "and a philosophy of intentional minimalism over decoration."
    ),
    "aesthetic": ["japandi", "wabi-sabi", "zen minimalist", "nordic calm"],
    "colors": ["warm white", "pale oak", "charcoal", "stone grey"],
    "materials": ["linen", "oak", "bamboo", "matte ceramic"],
    "budget_max": 150.0,
    "budget_currency": "USD",
}

STYLE_Y2K = {
    "style_profile": (
        "Nostalgic Y2K / early 2000s revival: iridescent and holographic "
        "surfaces, bubblegum pinks, chrome accents, low-rise silhouettes, "
        "butterfly motifs, and playful maximalism with a futuristic edge."
    ),
    "aesthetic": ["Y2K", "cyber Y2K", "2000s nostalgia", "pop maximalist"],
    "colors": ["bubblegum pink", "chrome silver", "electric blue", "white"],
    "materials": ["holographic vinyl", "patent leather", "mesh", "acrylic"],
    "budget_max": 60.0,
    "budget_currency": "USD",
}

STYLES = {
    "cottagecore": STYLE_COTTAGECORE,
    "japandi":     STYLE_JAPANDI,
    "y2k":         STYLE_Y2K,
}

REQUIRED_PRODUCT_KEYS = {"image_url", "product_name", "price", "link", "tags"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_env():
    missing = [k for k in ("SERPAPI_API_KEY", "HF_TOKEN") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in .env or the environment."
        )


def _assert_product_shape(items: list, label: str = ""):
    assert isinstance(items, list), f"[{label}] procurement_products must be a list"
    assert len(items) > 0, f"[{label}] procurement_products must not be empty"
    for it in items:
        assert set(it.keys()) == REQUIRED_PRODUCT_KEYS, (
            f"[{label}] Unexpected product keys: {set(it.keys())}"
        )
        assert isinstance(it["image_url"], str)
        assert isinstance(it["product_name"], str)
        assert it["price"] is None or isinstance(it["price"], float)
        assert isinstance(it["link"], str)
        assert " " not in it["link"], f"[{label}] URL contains spaces: {it['link']}"
        assert isinstance(it["tags"], list)


def _run_style(stylist_output: dict, critic_feedback: str = None) -> dict:
    state = {
        "stylist_output": stylist_output,
        "num_queries": 3,
        "results_per_query": 5,  # keep low for test speed
    }
    if critic_feedback:
        state["critic_feedback"] = critic_feedback
    return json.loads(run(state))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_styles_first_pass():
    """
    Runs all three styles through the procurement agent with no critic feedback.
    Verifies each returns LLM-generated queries and a valid candidate pool.
    """
    _check_env()

    for name, style in STYLES.items():
        print(f"\n{'='*60}")
        print(f"Style: {name.upper()}")
        print(f"{'='*60}")

        result = _run_style(style)

        queries = result.get("procurement_queries", [])
        print(f"Queries:\n" + json.dumps(queries, indent=2))
        assert isinstance(queries, list), f"[{name}] queries must be a list"
        assert len(queries) == 3, f"[{name}] expected 3 queries, got {len(queries)}"
        assert all(isinstance(q, str) and q.strip() for q in queries), (
            f"[{name}] all queries must be non-empty strings"
        )

        assert result.get("style_profile") == style["style_profile"], (
            f"[{name}] style_profile must pass through unchanged"
        )

        items = result.get("procurement_products", [])
        print(f"Total candidates: {len(items)}")
        # Products are appended in query order — stride by results_per_query
        # to approximate the first result from each query's bucket.
        stride = 5  # matches results_per_query in _run_style
        for i, q in enumerate(queries):
            sample = items[i * stride] if i * stride < len(items) else None
            print(f"\n  [{i+1}] Query: {q}")
            print(f"       Sample: {json.dumps(sample, indent=2)}")
        _assert_product_shape(items, label=name)


def test_cottagecore_with_critic_feedback():
    """
    Retry-pass test using the cottagecore style.
    Verifies the agent returns a valid pool when critic_feedback is present.
    """
    _check_env()

    result = _run_style(
        STYLE_COTTAGECORE,
        critic_feedback=(
            "Outdoor lighting results scored 0.28 avg — too industrial/modern. "
            "Garden decor results scored 0.31 avg — too generic. "
            "Planters scored 0.79 avg — keep the same approach."
        ),
    )

    queries = result.get("procurement_queries", [])
    print("\nQueries (with critic feedback):\n" + json.dumps(queries, indent=2))
    assert len(queries) == 3
    assert all(isinstance(q, str) and q.strip() for q in queries)

    items = result.get("procurement_products", [])
    print(f"Total candidates: {len(items)}")
    _assert_product_shape(items, label="cottagecore_retry")


def test_japandi_with_critic_feedback():
    """
    Retry-pass test using the japandi style.
    """
    _check_env()

    result = _run_style(
        STYLE_JAPANDI,
        critic_feedback=(
            "Furniture results scored 0.25 avg — too ornate, not minimal enough. "
            "Textiles scored 0.71 avg — good, keep this angle. "
            "Ceramics scored 0.38 avg — too decorative, need plainer forms."
        ),
    )

    queries = result.get("procurement_queries", [])
    print("\nQueries (with critic feedback):\n" + json.dumps(queries, indent=2))
    assert len(queries) == 3
    assert all(isinstance(q, str) and q.strip() for q in queries)

    items = result.get("procurement_products", [])
    print(f"Total candidates: {len(items)}")
    _assert_product_shape(items, label="japandi_retry")


if __name__ == "__main__":
    test_all_styles_first_pass()
    print("\n✓ test_all_styles_first_pass passed")

    test_cottagecore_with_critic_feedback()
    print("\n✓ test_cottagecore_with_critic_feedback passed")

    test_japandi_with_critic_feedback()
    print("\n✓ test_japandi_with_critic_feedback passed")
