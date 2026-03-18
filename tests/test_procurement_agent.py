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


MOCK_STYLIST_OUTPUT = {
    "style_profile": (
        "Bright spring garden vibe: airy pastels, natural textures, "
        "cottagecore-meets-modern planters, light wood accents."
    ),
    "keywords": ["spring garden", "planter", "outdoor decor", "patio", "string lights"],
    "colors": ["sage green", "cream", "terracotta"],
    "materials": ["ceramic", "rattan", "wood"],
    "categories": ["planters", "outdoor lighting", "garden decor"],
    "budget_max": 75.0,
    "budget_currency": "USD",
}


def test_procurement_returns_trimmed_json_shape():
    if not os.environ.get("SERPAPI_API_KEY"):
        raise RuntimeError("SERPAPI_API_KEY is required. Set it in .env or environment.")

    result_json = run(
        {
            "num_products": 10,
            "stylist_output": MOCK_STYLIST_OUTPUT,
        }
    )

    result = json.loads(result_json)
    print("style_profile:\n" + result.get("style_profile", ""))
    print("\nprocurement_queries:\n" + json.dumps(result.get("procurement_queries", []), indent=2))
    assert "procurement_products" in result
    items = result["procurement_products"]
    print("\nprocurement_products sample:\n" + json.dumps(items, indent=2))
    assert isinstance(items, list)
    assert len(items) > 0

    for it in items:
        assert set(it.keys()) == {"image_url", "product_name", "price", "link", "tags"}
        assert isinstance(it["image_url"], str)
        assert isinstance(it["product_name"], str)
        assert it["price"] is None or isinstance(it["price"], float)
        assert isinstance(it["link"], str)
        assert " " not in it["link"], f"URL contains spaces: {it['link']}"


if __name__ == "__main__":
    test_procurement_returns_trimmed_json_shape()
    print("test_procurement_returns_trimmed_json_shape passed")
