# VisionCart — Procurement Agent Guide
`src/agents/procurement.py`

---

## What this agent does

The procurement agent is the second step in the VisionCart pipeline. Given the stylist agent's style profile, it determines which product categories suit the aesthetic, generates one search query per category via LLM, and fetches real purchasable products from Google Shopping.

1. **Reads the style profile and optional critic feedback from state** — parses `stylist_output` into a typed `StylistOutput` object. If `critic_feedback` is present (set by the critic agent on a retry pass), it is forwarded to the query builder.
2. **Generates queries with one LLM call** — sends the full style profile, colors, materials, aesthetics, budget, and any critic feedback to an LLM. The model decides which product categories suit the aesthetic and returns one specific, context-aware query per category.
3. **Fires all queries and collects results** — each query fetches up to `results_per_query` results from SerpAPI. A global dedup set (keyed on product URL + normalized name) ensures no product appears more than once across queries.
4. **Passes the full candidate pool to the ranker** — no trimming or round-robin is applied. The ranker and critic decide the final selection.
5. **Returns a JSON string** — containing all candidates, the queries used, and the narrative style profile.

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

Google Shopping already ranks its own results by relevance to the query — the first results are the strongest signal, and quality degrades beyond ~20–25. Fetching more than ~25 results brings in progressively generic products that the ranker will score low anyway, at the cost of extra inference time. Each SerpAPI request costs 1 credit regardless of `results_per_query`.

### `critic_feedback` (optional)

String set by the critic agent on a retry pass. Describes which categories scored poorly against the style embeddings and why. When present, the LLM uses this to generate different query angles for the failing categories.

Example value written by the critic:
```
"Retrieved 87 products, accepted 2, rejected 85.
Acceptance rate was low (2%). Queries were too broad — use more specific aesthetic and material terms.
Style signals absent from all accepted products: herringbone, tweed, scholarly.
Try queries targeting: herringbone, tweed, scholarly."
```

---

## Output: Product Candidates

The agent returns a **JSON string** (not a dict) with three top-level keys. The caller (`main-LC.py` or `procurement_node` in `main-LG.py`) is responsible for parsing it with `json.loads()`.

### `procurement_queries`
**Type:** `list[str]`

LLM-generated queries — one per category the model chose to explore.

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

The free-text style description passed through from the stylist agent. Note: this is the raw narrative string, not the structured dict the ranker needs. The orchestrators (`main-LC.py`, `main-LG.py`) call `_style_profile_from_stylist()` to build the structured version separately.

---

### `procurement_products`
**Type:** `list[dict]`

Deduplicated list of all product candidates from SerpAPI across all queries. Products are deduplicated by `(product_url, normalized_product_name)`.

| Key | Type | Description |
|-----|------|-------------|
| `product_name` | `str` | Product title as returned by Google Shopping |
| `image_url` | `str` | URL to the product thumbnail image |
| `price` | `float \| null` | Parsed numeric price; `null` if unparseable |
| `link` | `str` | URL to the product page |
| `tags` | `list[str]` | Keywords extracted from the product title |

Product IDs are **not** generated by this agent. The ranker generates fallback IDs (`p_{idx}`) for any product missing a `product_id`.

---

## Query generation and LLM backend

`build_queries()` selects an LLM backend based on environment variables:

| Condition | Backend |
|-----------|---------|
| `OLLAMA_MODEL` env var is set | Local Ollama (model from `OLLAMA_MODEL`) |
| `OLLAMA_MODEL` not set, `HF_TOKEN` not set | Local Ollama with default model (`qwen2.5:7b`) |
| `OLLAMA_MODEL` not set, `HF_TOKEN` is set | HuggingFace Inference API (Qwen/Qwen2.5-7B-Instruct) |

Override the Ollama host with `OLLAMA_HOST` (default: `http://localhost:11434`).

The prompt includes the full style profile and any critic feedback:

```
Given a style profile, decide which product categories a shopper would want to buy
to achieve this aesthetic, then generate one search query per category.

Style profile: Bright spring garden vibe: airy pastels...
Colors: sage green, cream, terracotta
Materials: ceramic, rattan, wood
Aesthetics: cottagecore, boho outdoor, spring patio, whimsical garden

[on retry] The previous search was evaluated and had these issues — address them:
  Acceptance rate was low (2%). Use more specific aesthetic and material terms.
  ...

Generate exactly 5 queries. Return a JSON array of strings and nothing else.
```

If the API call fails (network error, bad response), a `RuntimeError` is raised — there is no silent fallback.

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:

```
requests>=2.28.0           # HTTP calls to SerpAPI and Ollama
python-dotenv>=1.0.0       # loads .env into os.environ
huggingface_hub>=0.23.0    # HuggingFace Inference API (optional)
```

### API keys

```bash
# .env
SERPAPI_API_KEY=your_serpapi_key_here
HF_TOKEN=your_huggingface_token_here   # optional if using Ollama
OLLAMA_MODEL=qwen2.5:7b               # optional; triggers local Ollama backend
```

`HF_TOKEN` is optional — if `OLLAMA_MODEL` is set or `HF_TOKEN` is absent, the agent uses local Ollama for query generation. `SERPAPI_API_KEY` is always required.

---

## Testing the agent in isolation

```bash
cd VisionCart
python tests/test_procurement_agent.py
```

To test the retry loop behavior, add `critic_feedback` to the mock state:

```python
import json
import src.agents.procurement as procurement

state = {
    "stylist_output": {
        "style_profile": "Bright spring garden vibe: airy pastels, natural textures...",
        "aesthetic": ["cottagecore", "boho outdoor"],
        "colors": ["sage green", "cream", "terracotta"],
        "materials": ["ceramic", "rattan", "wood"],
    },
    "num_queries": 5,
    "results_per_query": 20,
    "critic_feedback": (
        "Acceptance rate was low (2%). Queries were too broad. "
        "Style signals absent from all accepted products: herringbone, tweed."
    ),
}
result = json.loads(procurement.run(state))
print(result["procurement_queries"])
print(f"Total candidates: {len(result['procurement_products'])}")
```

The queries returned on this pass should visibly differ from a first-pass run with no feedback.
