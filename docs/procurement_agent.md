# VisionCart — Procurement Agent Guide
`src/agents/procurement.py`

---

## What this agent does

The procurement agent is the second step in the VisionCart pipeline. It takes the stylist agent's style profile and queries SerpAPI's Google Shopping engine to find real products that match the aesthetic.

**Inputs it receives** (from `state`):
- `state["stylist_output"]` — the style profile produced by `stylist.py` as a dict
- `state["serpapi_api_key"]` — SerpAPI key (or set `SERPAPI_API_KEY` env var)
- `state["num_products"]` — how many products to return (default: 10)

**Output it produces:**
- A JSON string containing the matched product candidates and the queries used

---

## Output JSON Reference

The agent returns a JSON string with three top-level keys.

### `procurement_queries`
**Type:** `list[str]`

The search queries sent to SerpAPI, built dynamically from the stylist output. The first query combines categories, materials, colors, and keywords into a base string. Each additional query appends a remaining keyword as a suffix to broaden recall.

```json
"procurement_queries": [
  "planters outdoor lighting ceramic sage green spring garden planter",
  "planters outdoor lighting ceramic sage green spring garden planter outdoor decor",
  "planters outdoor lighting ceramic sage green spring garden planter patio",
  "planters outdoor lighting ceramic sage green spring garden planter string lights"
]
```

---

### `style_profile`
**Type:** `str`

The free-text style description from the stylist agent, passed through for use by downstream agents (ranker, output).

```json
"style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern planters, light wood accents."
```

---

### `procurement_products`
**Type:** `list[object]`

Deduplicated list of product candidates from SerpAPI, trimmed to `num_products`. Each item has the following keys:

#### `product_name`
**Type:** `str`

The product title as returned by Google Shopping.

```json
"product_name": "Rattan Planter Basket with Drainage Hole"
```

#### `image_url`
**Type:** `str`

URL to the product thumbnail image.

```json
"image_url": "https://encrypted-tbn1.gstatic.com/shopping?q=..."
```

#### `price`
**Type:** `float | null`

Parsed numeric price. `null` if the price string could not be parsed (e.g. "Contact for price").

```json
"price": 24.99
```

#### `link`
**Type:** `str`

Percent-encoded URL to the product page on the merchant's site. Spaces and special characters are encoded so the URL is always valid.

```json
"link": "https://www.example.com/products/rattan-planter-basket"
```

#### `tags`
**Type:** `list[str]`

Descriptive keyword tags extracted from the product title by splitting on delimiters (`-`, `,`, `|`, `/`, etc.) and filtering out stop words and digits. These reflect the material, style, and category as described by the merchant — useful for the ranker to match against the style profile.

```json
"tags": ["rattan planter basket", "drainage hole", "boho home decor"]
```

---

## Input State Reference

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `stylist_output` | `dict` | Yes | Style profile from the stylist agent |
| `serpapi_api_key` | `str` | No | Falls back to `SERPAPI_API_KEY` env var |
| `num_products` | `int` | No | Number of products to return (default: `10`) |

### `stylist_output` keys

| Key | Type | Description |
|-----|------|-------------|
| `style_profile` | `str` | Free-text style description |
| `keywords` | `list[str]` | Style/product keywords |
| `colors` | `list[str]` | Color palette |
| `materials` | `list[str]` | Material types |
| `categories` | `list[str]` | Product categories |
| `budget_max` | `float` | Optional max budget |
| `budget_currency` | `str` | Optional currency code (e.g. `"USD"`) |

---

## Mock Stylist Output

Used in `tests/test_procurement_agent.py` as a stand-in for real stylist agent output. Pass this as `stylist_output` in state when testing procurement in isolation.

```json
{
  "style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern planters, light wood accents.",
  "keywords": ["spring garden", "planter", "outdoor decor", "patio", "string lights"],
  "colors": ["sage green", "cream", "terracotta"],
  "materials": ["ceramic", "rattan", "wood"],
  "categories": ["planters", "outdoor lighting", "garden decor"],
  "budget_max": 75.0,
  "budget_currency": "USD"
}
```

---

## Testing the agent in isolation

```python
# tests/test_procurement_agent.py

import sys, json, os
sys.path.insert(0, "src")

from agents.procurement import run

MOCK_STYLIST_OUTPUT = {
    "style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern planters, light wood accents.",
    "keywords": ["spring garden", "planter", "outdoor decor", "patio", "string lights"],
    "colors": ["sage green", "cream", "terracotta"],
    "materials": ["ceramic", "rattan", "wood"],
    "categories": ["planters", "outdoor lighting", "garden decor"],
    "budget_max": 75.0,
    "budget_currency": "USD",
}

result_json = run({
    "serpapi_api_key": os.environ["SERPAPI_API_KEY"],
    "num_products": 10,
    "stylist_output": MOCK_STYLIST_OUTPUT,
})

print(result_json)
```

Run it with:
```bash
cd VisionCart
python tests/test_procurement_agent.py
```

---

## Connecting to the LangGraph graph

```python
# graph/state.py (rough example)

from langgraph.graph import StateGraph
from agents import stylist, procurement, ranker, critic, output

graph = StateGraph(dict)
graph.add_node("stylist", stylist.run)
graph.add_node("procurement", procurement.run)  # <-- this agent
graph.add_node("ranker", ranker.run)
graph.add_node("critic", critic.run)
graph.add_node("output", output.run)

graph.set_entry_point("stylist")
graph.add_edge("stylist", "procurement")
graph.add_edge("procurement", "ranker")
graph.add_edge("ranker", "critic")
graph.add_edge("critic", "output")
graph.set_finish_point("output")
```
