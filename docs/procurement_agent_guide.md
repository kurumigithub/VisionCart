# VisionCart — Procurement Agent Guide
`src/agents/procurement.py`

---

## What this agent does

The procurement agent is the second step in the VisionCart pipeline. Given the stylist agent's style profile, it determines which product categories suit the aesthetic, generates one search query per category via LLM, and fetches real purchasable products from Google Shopping.

1. **Reads the style profile and optional critic feedback from state** — parses `stylist_output` into a typed `StylistOutput` object. If `critic_feedback` is present (set by the critic agent on a retry pass), it is forwarded to the query builder.
2. **Generates queries with one LLM call** — sends the full style profile, colors, materials, aesthetics, budget, and any critic feedback to **Qwen2.5-7B-Instruct** via the HuggingFace Inference API. The model decides which product categories suit the aesthetic and returns one specific, context-aware query per category. If the API call fails for any reason, a `RuntimeError` is raised immediately — there is no silent fallback.
3. **Fires all queries and collects per-query buckets** — each query fetches up to `num_products` results from SerpAPI. A global dedup set ensures no product appears in more than one bucket.
4. **Round-robins across buckets to fill the output** — takes one result from each query bucket in turn so every category contributes evenly to the final list, regardless of which query had the most results.
5. **Trims and shapes the output** — cuts the merged list to `num_products`, strips each item to only the fields downstream agents need (`product_name`, `image_url`, `price`, `link`, `tags`), and returns everything as a JSON string.

---

## Input: State keys

### `stylist_output` (required)

The dict produced by `stylist.py`. The stylist detects vibes, aesthetics, colors, and materials only — it does **not** identify purchasable product categories. The procurement LLM determines what to search for.

```json
{
  "style_profile": "Bright spring garden vibe: airy pastels, natural textures, cottagecore-meets-modern sensibility, light wood accents, and a relaxed outdoor-living feeling.",
  "aesthetic": ["cottagecore", "boho outdoor", "spring patio", "whimsical garden"],
  "colors": ["sage green", "cream", "terracotta"],
  "materials": ["ceramic", "rattan", "wood"],
  "budget_max": 75.0,
  "budget_currency": "USD"
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `style_profile` | `str` | Yes | Full narrative prose describing the aesthetic — fed directly into the LLM query prompt |
| `aesthetic` | `list[str]` | Yes | Mood/vibe descriptors (e.g. `"cottagecore"`) — style context for the LLM |
| `colors` | `list[str]` | Yes | Color palette |
| `materials` | `list[str]` | Yes | Material types |
| `budget_max` | `float` | No | Optional max budget, included in the LLM prompt |
| `budget_currency` | `str` | No | Optional currency code (e.g. `"USD"`) |

### `num_queries` (optional)

Integer, default `5`. Controls how many search queries (i.e. product categories) the LLM is asked to generate. Increase for broader exploration, decrease for faster runs.

### `results_per_query` (optional)

Integer, default `20`. Controls how many results to fetch from SerpAPI per query.

Google Shopping already ranks its own results by relevance to the query — the first results are the strongest signal, and quality degrades beyond ~20–25. Fetching more than ~25 results brings in progressively generic junk that the ranker will score low anyway, at the cost of extra CLIP inference time. The default of 20 captures the relevant top of each query's result set without pulling in noise.

Each SerpAPI request costs 1 credit regardless of `results_per_query`, so the only cost to increasing this is ranker latency.

### `critic_feedback` (optional)

String set by the critic agent on a retry pass. Describes which categories scored poorly against the style embeddings and why. When present, the LLM uses this to generate different query angles for the failing categories.

Example value written by the critic:
```
"Outdoor lighting results scored 0.28 avg — too industrial/modern in style.
Garden decor results scored 0.31 avg — too generic, no aesthetic specificity.
Planters scored 0.79 avg — keep the same approach for this category."
```

---

## Output: Product Candidates

The agent returns a JSON string with three top-level keys.

### `procurement_queries`
**Type:** `list[str]`

LLM-generated queries — one per category the model chose to explore. The model decides both which categories to cover and what angle to use for each query.

First-pass example:
```json
"procurement_queries": [
  "handmade terracotta ceramic planter cottagecore artisan",
  "rattan wicker outdoor string lights warm boho patio",
  "vintage ceramic garden ornament whimsical cottage floral",
  "woven jute doormat natural fiber boho spring",
  "linen garden apron sage green cottagecore"
]
```

Retry-pass example (after critic feedback on lighting and garden decor):
```json
"procurement_queries": [
  "handmade terracotta ceramic planter cottagecore artisan",
  "solar fairy lights garden cottagecore warm soft glow",
  "botanical ceramic garden stake hand-painted spring floral",
  "woven jute doormat natural fiber boho spring",
  "linen garden apron sage green cottagecore"
]
```

---

### `style_profile`
**Type:** `str`

The free-text style description passed through from the stylist agent for use by downstream agents (ranker, output).

---

### `procurement_products`
**Type:** `list[object]`

Deduplicated list of product candidates from SerpAPI, trimmed to `num_products`, merged via round-robin across query buckets.

| Key | Type | Description |
|-----|------|-------------|
| `product_name` | `str` | Product title as returned by Google Shopping |
| `image_url` | `str` | URL to the product thumbnail image — used by the ranker for CLIP embeddings |
| `price` | `float \| null` | Parsed numeric price; `null` if unparseable |
| `link` | `str` | URL to the product page |
| `tags` | `list[str]` | Keywords extracted from the product title — used by the ranker for text similarity |

---

## Query generation

`build_queries()` sends a single structured prompt to **Qwen/Qwen2.5-7B-Instruct** via the HuggingFace Inference API, containing the full style profile and any critic feedback. The model decides which product categories suit the aesthetic and generates one query per category.

Example prompt (abbreviated):
```
Given a style profile, decide which product categories a shopper would want to buy
to achieve this aesthetic, then generate one search query per category.

Style profile: Bright spring garden vibe: airy pastels, natural textures...
Colors: sage green, cream, terracotta
Materials: ceramic, rattan, wood
Aesthetics: cottagecore, boho outdoor, spring patio, whimsical garden

[on retry] The previous search had these issues:
  Outdoor lighting results scored 0.28 avg — too industrial/modern in style.
  ...

Generate exactly 5 queries. Return a JSON array of strings and nothing else.
```

If `HF_TOKEN` is not set or the API call fails for any reason, the agent raises a `RuntimeError` — there is no silent fallback to a deterministic formula.

---

## Setup

### Install dependencies

All packages this agent needs are already in `requirements.txt`:

```
requests>=2.28.0           # HTTP calls to SerpAPI
python-dotenv>=1.0.0       # loads .env into os.environ
huggingface_hub>=0.23.0    # HuggingFace Inference API (Qwen2.5-7B-Instruct)
```

Install with:
```bash
pip install -r requirements.txt
```

### API keys

The agent reads two keys from the environment (or `.env` at the repo root):

```bash
# .env
SERPAPI_API_KEY=your_serpapi_key_here
HF_TOKEN=your_huggingface_token_here
```

Both keys are required. Get a free `HF_TOKEN` at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **read** access, no paid plan needed.

---

## Testing the agent in isolation

```bash
cd VisionCart
python tests/test_procurement_agent.py
```

The mock stylist output should no longer include a `products` field — only `style_profile`, `aesthetic`, `colors`, `materials`, and optionally `budget_max`/`budget_currency`.

To test the retry loop behavior, add `critic_feedback` to the mock state:

```python
state = {
    "stylist_output": MOCK_STYLIST_OUTPUT,
    "num_queries": 5,
    "results_per_query": 20,
    "critic_feedback": (
        "Outdoor lighting results scored 0.28 avg — too industrial. "
        "Garden decor results scored 0.31 avg — too generic."
    ),
}
result = json.loads(procurement.run(state))
```

The queries returned on this pass should visibly differ from a first-pass run with no feedback.
