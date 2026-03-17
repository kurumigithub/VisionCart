import sys
sys.path.insert(0, "src")

from agents.output import run, build_output_products

# --- mock state -----------------------------------------------------------

MOCK_STATE = {
    "style_profile": (
        "Warm earth tones, relaxed Japanese streetwear silhouettes, "
        "natural textures like linen and washed denim, minimal branding."
    ),
    "ranked_products": [
        {
            "name": "Uniqlo Wide Linen Trousers",
            "score": 0.91,
            "tags": ["linen", "wide-leg", "neutral", "minimalist"],
            "url": "https://uniqlo.com/example",
            "image_url": "https://uniqlo.com/example.jpg",
            "price": "$49.90",
        },
        {
            "name": "Mango Washed Denim Overshirt",
            "score": 0.84,
            "tags": ["denim", "oversized", "earth tone", "texture"],
            "url": "https://mango.com/example",
            "image_url": "https://mango.com/example.jpg",
            "price": "$69.99",
        },
        {
            "name": "COS Relaxed Cotton Jacket",
            "score": 0.78,
            "tags": ["cotton", "boxy", "minimalist", "neutral"],
            "url": "https://cos.com/example",
            "image_url": "",
            "price": "$129.00",
        },
    ],
    "critic_notes": "Filtered 2 products that matched color but had prominent logo branding.",
}

# --- tests ----------------------------------------------------------------

def test_build_output_products():
    products = build_output_products(MOCK_STATE["ranked_products"])
    assert len(products) == 3
    for p in products:
        assert "name" in p
        assert "price" in p
        assert "url" in p
        assert "image_url" in p
        assert "score" in p
        assert "tags" in p
    print("test_build_output_products passed")


def test_run_returns_expected_keys():
    result = run(MOCK_STATE)
    assert "output_text" in result, "missing output_text"
    assert "output_products" in result, "missing output_products"
    assert isinstance(result["output_text"], str)
    assert len(result["output_text"]) > 0
    assert isinstance(result["output_products"], list)
    assert len(result["output_products"]) == 3
    print("test_run_returns_expected_keys passed")


def test_run_empty_products():
    result = run({**MOCK_STATE, "ranked_products": []})
    assert "output_text" in result
    assert "output_products" not in result  # early return, no products built
    print("test_run_empty_products passed")


def test_run_empty_style_profile():
    result = run({**MOCK_STATE, "style_profile": ""})
    assert "output_text" in result
    print("test_run_empty_style_profile passed")


if __name__ == "__main__":
    print("Running unit tests (no model)...")
    test_build_output_products()
    test_run_empty_products()
    test_run_empty_style_profile()

    print("\nRunning full inference test (loads Qwen model)...")
    test_run_returns_expected_keys()

    print("\n--- output_text ---")
    result = run(MOCK_STATE)
    print(result["output_text"])

    print("\n--- output_products ---")
    for p in result["output_products"]:
        print(p)
