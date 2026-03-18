# VisionCart — Procurement Agent Guide
`src/agents/procurement.py`

---

## What this agent does

The procurement agent is the second step in the VisionCart pipeline. Given the stylist agent's structured style profile, it finds real purchasable products that match the aesthetic.

1. **Reads the style profile from state** — parses `stylist_output` into a typed `StylistOutput` object with `products`, `materials`, `colors`, and `aesthetic`.
2. **Builds one query per product type** — each query = product + random material + random color + random aesthetic descriptor. Query count equals the number of product types the stylist detected, so API calls scale naturally with the richness of the style board.
3. **Fires all queries and collects per-query buckets** — each query fetches up to `num_products` results from SerpAPI. A global dedup set ensures no product appears in more than one bucket.
4. **Round-robins across buckets to fill the output** — takes one result from each query bucket in turn so every product type contributes evenly to the final list, regardless of which query had the most results.
5. **Trims and shapes the output** — cuts the merged list to `num_products`, strips each item to only the fields downstream agents need (`product_name`, `image_url`, `price`, `link`, `tags`), and returns everything as a JSON string.

---

## Input: Stylist Output

The agent receives `state["stylist_output"]` — the dict produced by `stylist.py`. Here's what that looks like:

```json
{
  "style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern sensibility, light wood accents, and a relaxed outdoor-living feeling.",
  "products": ["planters", "outdoor lighting", "garden decor"],
  "aesthetic": ["cottagecore", "boho outdoor", "spring patio", "whimsical garden"],
  "colors": ["sage green", "cream", "terracotta"],
  "materials": ["ceramic", "rattan", "wood"],
  "budget_max": 75.0,
  "budget_currency": "USD"
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `style_profile` | `str` | Yes | Full narrative prose describing the aesthetic — for downstream LLM agents (ranker, critic), never used as a search query |
| `products` | `list[str]` | Yes | Purchasable item types to search for (e.g. `"planters"`) — drives query count (1 query per product) |
| `aesthetic` | `list[str]` | Yes | Mood/vibe/style-movement descriptors, NOT product names (e.g. `"cottagecore"`) — randomly appended to queries |
| `colors` | `list[str]` | Yes | Color palette |
| `materials` | `list[str]` | Yes | Material types |
| `budget_max` | `float` | No | Optional max budget |
| `budget_currency` | `str` | No | Optional currency code (e.g. `"USD"`) |

The agent also reads `state["num_products"]` (int, default `10`) to control how many products to return.

---

## Output: Product Candidates

The agent returns a JSON string with three top-level keys.

### `procurement_queries`
**Type:** `list[str]`

One query per product type — each a randomized combination of product + material + color + aesthetic. Example run with the spring garden input:

```json
"procurement_queries": [
  "planters rattan sage green cottagecore",
  "outdoor lighting ceramic cream spring patio",
  "garden decor wood terracotta boho outdoor"
]
```

Each run produces different queries since material, color, and aesthetic are randomly sampled.

---

### `style_profile`
**Type:** `str`

The free-text style description passed through from the stylist agent for use by downstream agents (ranker, output).

```json
"style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern sensibility, light wood accents, and a relaxed outdoor-living feeling."
```

---

### `procurement_products`
**Type:** `list[object]`

Deduplicated list of product candidates from SerpAPI, trimmed to `num_products`, merged via round-robin across query buckets.

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

- **Mock input** — modify `MOCK_STYLIST_OUTPUT` at the top of the file to change the style profile, products, aesthetic, colors, or materials fed into the agent.
- **Number of products** — change `"num_products": 10` inside the `run(...)` call in `test_procurement_returns_trimmed_json_shape()`.
