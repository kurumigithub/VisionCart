# VisionCart — Ranker / Critic Agent Guide
`src/agents/ranker_critic.py`

---

## What this agent does

The ranker/critic agent is the third step in the VisionCart pipeline. It receives the raw product pool from procurement and the style profile from the stylist, scores every product against the user's aesthetic, and filters out poor matches before passing the results to the output agent.

Two roles are combined into one agent deliberately — scoring and filtering are tightly coupled and separating them would require passing intermediate state that neither agent would own cleanly.

1. **Reads the style profile and candidate products from state** — accepts both `candidate_products` (canonical key) and `procurement_products` (procurement's output key) so it works whether called directly or after procurement in the pipeline.
2. **Scores each product on three dimensions** — image similarity (cosine distance between embeddings), text similarity (keyword overlap), and semantic match (rule-based material/color/style alignment).
3. **Critic evaluation** — applies hard rejection rules before finalizing scores: must-avoid term violations, image similarity floor, and overall weighted score threshold.
4. **Sorts and ranks accepted products** — accepted products are sorted by final score descending and assigned integer ranks starting from 1.
5. **Writes ranked and rejected products to state** — output agent receives the ranked list; `critic_feedback` is generated for procurement on a retry pass.

---

## Input: State keys

### `style_profile` (required)

A structured dict describing the user's aesthetic. Built by `_style_profile_from_stylist()` in both orchestrators before this agent runs.

```python
state["style_profile"] = {
    "board_id": "dark_academia__blazer_query",
    "style_summary": "Scholarly, vintage-inspired aesthetic with rich earth tones.",
    "style_keywords": ["herringbone", "tweed", "blazer", "leather", "vintage"],
    "style_elements": ["tailored", "structured", "dark", "layered"],
    "color_palette": {
        "dominant": ["brown", "navy", "forest green", "burgundy"],
        "accent": ["gold", "cream", "camel"],
        "avoid": []
    },
    "materials": {
        "preferred": ["wool", "tweed", "leather", "velvet"],
        "avoid": []
    },
    "constraints": {
        "must_avoid": []
    },
    "board_embedding": []   # list of floats if image embeddings available; [] for text-only mode
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `style_keywords` | `list[str]` | Yes | Aesthetic and material terms — primary signal for text similarity scoring |
| `style_elements` | `list[str]` | No | Secondary descriptors — added to keyword pool for semantic scoring |
| `color_palette.dominant` | `list[str]` | No | Preferred dominant colors — each match adds `+0.10` to semantic score |
| `color_palette.accent` | `list[str]` | No | Accent colors — each match adds `+0.05` to semantic score |
| `color_palette.avoid` | `list[str]` | No | Colors to penalize — each occurrence subtracts `0.25` from semantic score |
| `materials.preferred` | `list[str]` | No | Preferred materials — each match adds `+0.10` to semantic score |
| `materials.avoid` | `list[str]` | No | Materials to penalize — each occurrence subtracts `0.25` |
| `constraints.must_avoid` | `list[str]` | No | Hard rejection terms — any product containing these is auto-rejected before scoring |
| `board_embedding` | `list[float]` | No | CLIP/SigLIP embedding of the vision board. If empty (`[]`), the agent runs in text-only mode |
| `style_summary` | `str` | No | Narrative description — words longer than 3 chars are added to the keyword pool |

### `candidate_products` or `procurement_products` (required)

The raw product list from procurement. The ranker accepts either key name.

```python
[
    {
        "product_id": "p_0",            # optional — auto-generated as "p_{idx}" if missing
        "product_name": "Loft Herringbone Relaxed Blazer",
        "image_url": "https://...",
        "image_embedding": [],          # list of floats if pre-computed; [] skips image scoring
        "tags": ["blazer", "wool", "herringbone", "brown"],
        "price": 89.99,
        "link": "https://..."
    },
    ...
]
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `product_name` | `str` | Yes | Used for text similarity and must-avoid checks |
| `tags` | `list[str]` | No | Keyword tags — combined with product name for all text scoring |
| `image_embedding` | `list[float]` | No | Pre-computed product image embedding. If empty, image similarity is skipped |
| `image_url` | `str` | No | Product image URL — procurement returns this but embeddings are not currently computed |
| `price` | `float\|null` | No | Passed through to output |
| `link` | `str` | No | Passed through to output |

### `config` (optional)

A dict of threshold and weight overrides. All ranker constants can be overridden per-run without changing the source file.

```python
state["config"] = {
    "weight_image_similarity": 0.50,       # default
    "weight_text_similarity": 0.30,        # default
    "weight_semantic_match": 0.20,         # default
    "image_similarity_reject_floor": 0.25, # default — only applied when image embeddings present
    "overall_reject_threshold": 0.35,      # default
    "avoid_penalty": 0.25,                 # default
    "preferred_bonus": 0.10,               # default
    "must_avoid_auto_reject": True,        # default
}
```

---

## Output: State keys

### `ranked_products`
**Type:** `list[dict]`

Accepted products sorted by `score` descending. This is what the output agent receives.

```python
[
    {
        "name": "Loft Women's Herringbone Relaxed Blazer",
        "score": 0.7241,
        "tags": ["blazer", "wool", "herringbone", "brown"],
        "price": 89.99,
        "url": "https://...",
        "image_url": "https://..."
    },
    ...
]
```

### `rejected_products`
**Type:** `list[dict]`

Products that failed critic evaluation, with a human-readable reason.

```python
[
    {"product_id": "p_14", "reason": "Rejected: overall score too low (0.20 < 0.245)."},
    {"product_id": "p_27", "reason": "Rejected: must-avoid constraint violated ('synthetic' found in product)."},
]
```

### `ranker_critic_output`
**Type:** `dict`

Full structured output including score breakdowns for every accepted product. Useful for debugging.

```python
{
    "board_id": "dark_academia__blazer_query",
    "accepted_products": [
        {
            "product_id": "p_0",
            "rank": 1,
            "final_score": 0.7241,
            "score_breakdown": {
                "image_similarity": 0.0,
                "text_similarity": 0.8134,
                "semantic_match_score": 0.7000
            },
            "reason": "Strong match (good keyword alignment, strong style/material fit)."
        }
    ],
    "rejected_products": [...]
}
```

### `critic_feedback`
**Type:** `str | None`

An actionable feedback string for the procurement agent's retry pass. `None` when results are satisfactory (acceptance rate ≥ 50% and at least 5 products accepted). When set, describes issues like:

- Low acceptance rate → queries were too broad
- Category concentration → all accepted products are the same type
- Style keyword gaps → key aesthetic signals not represented in any accepted product

Example:
```
"Retrieved 87 products, accepted 2, rejected 85.
Acceptance rate was low (2%). Queries were too broad — use more specific aesthetic and material terms.
Style signals absent from all accepted products: herringbone, tweed, scholarly.
Try queries targeting: herringbone, tweed, scholarly."
```

---

## Scoring logic

### Three-dimensional scoring

Each product is scored on three axes:

| Dimension | Default weight | Method |
|-----------|---------------|--------|
| Image similarity | 50% | Cosine similarity between `board_embedding` and `product["image_embedding"]` |
| Text similarity | 30% | `keyword_overlap_score(style_keywords + style_summary_words, product_text)` |
| Semantic match | 20% | Rule-based: bonuses for preferred materials/colors, penalties for avoided ones |

**Text-only fallback** — when `board_embedding` or `product["image_embedding"]` is empty, image similarity is skipped and the remaining weights are renormalized:

```
final = (0.30 / 0.50) × txt_sim + (0.20 / 0.50) × sem_score
      = 0.60 × txt_sim + 0.40 × sem_score
```

This is the active code path in the current pipeline because procurement returns `image_url` strings only — no embeddings are computed.

### Semantic match scoring

Starts at a baseline of `0.50` for every product and moves up or down based on rule matches:

```
+0.10  per preferred material found in product text
+0.10  per dominant color found in product text
+0.05  per accent color found in product text
+0.20 × keyword_hit_rate  (style_keywords + style_elements)
-0.25  per avoid-list violation (color or material)
```

Score is clamped to `[0.0, 1.0]`.

---

## Critic rejection rules

Applied in order before the final score is calculated:

1. **Must-avoid auto-reject** — if any term from `constraints.must_avoid` appears anywhere in the product text, the product is immediately rejected with no score computed.
2. **Image similarity floor** — if image embeddings are present and `image_similarity < 0.25`, the product is rejected. Not applied in text-only mode.
3. **Avoid-list multi-violation** — if 2 or more avoid-list terms (colors or materials) are found, the product is rejected regardless of score.
4. **Overall score threshold** — if the weighted final score is below the effective threshold, the product is rejected.

In text-only mode, the overall reject threshold is automatically scaled: `effective_threshold = overall_reject_threshold × 0.7` (default `0.35 × 0.7 = 0.245`). This compensates for the fact that the original 0.35 threshold was calibrated with 50% image weight — without image scores, zero-keyword-overlap products would score `0.40 × 0.5 = 0.20`, always below 0.35.

---

## Internal helpers

### `_match_originals(accepted, candidates)`

Maps each `AcceptedProduct` back to its original candidate dict from procurement, using `product_id` as the key. This lets the ranker attach original fields (`product_name`, `image_url`, `price`, `link`, `tags`) to the ranked output without storing duplicates in `AcceptedProduct`.

### `_generate_critic_feedback(accepted, rejected, accepted_originals, style_profile)`

Builds the `critic_feedback` string by analyzing:
- **Acceptance rate** — if below 30%, signals queries were too broad
- **Category concentration** — if >60% of accepted products share the same product type (chair, dress, bag, etc.), suggests diversifying queries
- **Style keyword gaps** — identifies `style_keywords` and `style_elements` that appear in no accepted product, then recommends targeting them

Returns `None` when results are satisfactory (acceptance rate ≥ 50% and at least 5 accepted products with no other issues).

---

## Known limitations

**Image embeddings are not computed anywhere in the current pipeline.** Procurement returns `image_url` strings; the ranker expects pre-computed embedding vectors. The 50% image similarity weight is effectively dead weight — the ranker always runs in text-only mode with the adjusted `0.245` threshold. To activate image scoring, embeddings would need to be computed (e.g. with CLIP or SigLIP) at procurement time or as a separate preprocessing step.

---

## Setup

No external API keys required. The ranker is entirely local — no model is loaded, no network calls are made. All scoring is implemented in pure Python using utilities from `src/utils/helper.py`.

```bash
pip install -r requirements.txt
```

---

## Testing the agent in isolation

```python
import sys
sys.path.insert(0, "src")

from agents.ranker_critic import rank_and_critique

style_profile = {
    "style_keywords": ["herringbone", "wool", "blazer", "leather", "vintage"],
    "style_elements": ["tailored", "dark", "layered"],
    "color_palette": {
        "dominant": ["brown", "navy", "burgundy"],
        "accent": ["gold", "cream"],
        "avoid": []
    },
    "materials": {
        "preferred": ["wool", "tweed", "leather"],
        "avoid": ["synthetic", "polyester"]
    },
    "constraints": {"must_avoid": []},
    "board_embedding": []
}

candidates = [
    {
        "product_id": "p_0",
        "product_name": "Loft Women's Herringbone Relaxed Blazer",
        "tags": ["blazer", "wool", "herringbone", "brown"],
        "image_embedding": [],
        "price": 89.99,
        "link": "https://example.com/blazer"
    },
    {
        "product_id": "p_1",
        "product_name": "Zara Neon Polyester Mini Dress",
        "tags": ["neon", "polyester", "mini", "club"],
        "image_embedding": [],
        "price": 49.99,
        "link": "https://example.com/dress"
    }
]

result = rank_and_critique(style_profile, candidates)
print(f"Accepted: {len(result.accepted_products)}")
print(f"Rejected: {len(result.rejected_products)}")
for p in result.accepted_products:
    print(f"  Rank {p.rank}: {p.product_id}  score={p.final_score}  {p.reason}")
for p in result.rejected_products:
    print(f"  REJECTED {p.product_id}: {p.reason}")
```

Or test via the full `run()` LangGraph interface:

```python
from agents.ranker_critic import run as ranker_run

state = {
    "style_profile": style_profile,
    "candidate_products": candidates,
}
result = ranker_run(state)
print(result["ranked_products"])
print(result["rejected_products"])
print(result["critic_feedback"])
```

Run with:
```bash
cd VisionCart
python tests/test_eval.py --gt-only --style dark_academia
```

The `--gt-only` mode runs the ranker against only the 4 known-correct products per query, so any recall below 1.0 is a pure threshold or scoring problem.

---

## Connecting to the pipeline

The ranker sits between procurement and output in both orchestrators:

```python
# main-LC.py (simplified)
state["style_profile"] = _style_profile_from_stylist(state["stylist_output"])
ranker_results = ranker_critic.run(state)   # reads style_profile + candidate_products
state.update(ranker_results)                # writes ranked_products, rejected_products,
                                            #        ranker_critic_output, critic_feedback
output_results = output.run(state)          # reads ranked_products
```

In the LangGraph version (`main-LG.py`), `critic_feedback` stays in state and is automatically picked up by `procurement.run()` on the next pass if `should_retry()` routes back to procurement.
