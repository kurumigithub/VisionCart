# VisionCart — Procurement Agent Guide
`src/agents/procurement.py`

---

## What this agent does

The procurement agent is the second step in the VisionCart pipeline. Given the stylist agent's structured style profile, it finds real purchasable products that match the aesthetic.

1. **Reads the style profile from state** — parses `stylist_output` into a typed `StylistOutput` object with categories, materials, colors, and keywords.
2. **Builds search queries** — constructs a base query from the top 2 categories, top material, top color, and top 2 keywords. Then generates one additional query per remaining keyword as a suffix, broadening the search coverage without repeating results.
3. **Queries SerpAPI Google Shopping** — runs each query against the Google Shopping engine and collects all returned product results.
4. **Deduplicates results** — merges results across all queries and removes duplicates keyed on product URL + normalized product name.
5. **Trims and shapes the output** — cuts the deduplicated list to `num_products`, strips each item down to only the fields downstream agents need (`product_name`, `image_url`, `price`, `link`, `tags`), and returns everything as a JSON string.

---

## Input: Stylist Output

The agent receives `state["stylist_output"]` — the dict produced by `stylist.py`. Here's what that looks like:

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

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `style_profile` | `str` | Yes | Free-text style description |
| `keywords` | `list[str]` | Yes | Style/product keywords |
| `colors` | `list[str]` | Yes | Color palette |
| `materials` | `list[str]` | Yes | Material types |
| `categories` | `list[str]` | Yes | Product categories |
| `budget_max` | `float` | No | Optional max budget |
| `budget_currency` | `str` | No | Optional currency code (e.g. `"USD"`) |

The agent also reads `state["num_products"]` (int, default `10`) to control how many products to return.

---

## Output: Product Candidates

The agent returns a JSON string with three top-level keys.

### `procurement_queries`
**Type:** `list[str]`

Search queries built from the stylist input and sent to SerpAPI. The first query combines categories, materials, colors, and keywords into a base string. Each additional query appends a remaining keyword as a suffix to broaden recall.

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

The free-text style description passed through from the stylist agent for use by downstream agents (ranker, output).

```json
"style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern planters, light wood accents."
```

---

### `procurement_products`
**Type:** `list[object]`

Deduplicated list of product candidates from SerpAPI, trimmed to `num_products`.

| Key | Type | Description |
|-----|------|-------------|
| `product_name` | `str` | Product title as returned by Google Shopping |
| `image_url` | `str` | URL to the product thumbnail image |
| `price` | `float \| null` | Parsed numeric price; `null` if unparseable |
| `link` | `str` | Percent-encoded URL to the product page |
| `tags` | `list[str]` | Keywords extracted from the product title by splitting on delimiters (`-`, `,`, `\|`, `/`, etc.) and filtering stop words — useful for the ranker to match against the style profile |

```json
"procurement_products": [
  {
    "product_name": "Rattan Planter Basket with Drainage Hole",
    "image_url": "https://encrypted-tbn1.gstatic.com/shopping?q=...",
    "price": 24.99,
    "link": "https://www.example.com/products/rattan-planter-basket",
    "tags": ["rattan planter basket", "drainage hole", "boho home decor"]
  }
]
```

---

## Setup

### Install dependencies

All packages this agent needs are already in `requirements.txt`:

```
requests>=2.28.0       # HTTP calls to SerpAPI
python-dotenv>=1.0.0   # loads .env into os.environ
```

Install with:
```bash
# pip
pip install -r requirements.txt

# conda (activate your env first)
conda activate your_env_name
pip install -r requirements.txt
```

### API key

The agent reads `SERPAPI_API_KEY` from the environment. The `.env` file at the repo root already contains this key — no extra setup needed as long as you have the file.

To create your own key, sign up at [serpapi.com](https://serpapi.com) and copy the API key from your dashboard.

If the `.env` file doesn't exist yet, create one at the repo root and add your key:
```bash
# .env  (create this file at VisionCart/.env)
SERPAPI_API_KEY=your_key_here
```

---

## Testing the agent in isolation

Run it with:
```bash
cd VisionCart
python tests/test_procurement_agent.py
```

To adjust what gets tested, edit `tests/test_procurement_agent.py` directly:

- **Mock input** — modify `MOCK_STYLIST_OUTPUT` at the top of the file to change the style profile, keywords, colors, materials, or categories fed into the agent.
- **Number of products** — change `"num_products": 10` inside the `run(...)` call in `test_procurement_returns_trimmed_json_shape()`.