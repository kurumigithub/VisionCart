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
5. **Writes ranked and rejected products to state** — output agent receives the ranked list; the rejection log is available for critic feedback to procurement on retry.

---

## Input: State keys

### `style_profile` (required)

A structured dict describing the user's aesthetic. This is **not** the same as `stylist_output` — it must be a ranker-compatible dict with the keys below. The transform from `stylist_output` to this format currently needs to happen in `main.py` before the ranker runs (see known bug in `eval_tasks_2026-04-20.md`).

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
| `style_elements` | `list[str]` | No | Secondary descriptors — added to keyword pool for overlap scoring |
| `color_palette.dominant` | `list[str]` | No | Preferred dominant colors — each match adds a `+0.10` bonus to semantic score |
| `color_palette.accent` | `list[str]` | No | Accent colors — each match adds a `+0.05` bonus |
| `color_palette.avoid` | `list[str]` | No | Colors to penalize — each occurrence subtracts `0.25` from semantic score |
| `materials.preferred` | `list[str]` | No | Preferred materials — each match adds `+0.10` to semantic score |
| `materials.avoid` | `list[str]` | No | Materials to penalize — each occurrence subtracts `0.25` |
| `constraints.must_avoid` | `list[str]` | No | Hard rejection terms — any product containing these is auto-rejected before scoring |
| `board_embedding` | `list[float]` | No | CLIP/SigLIP embedding of the vision board. If empty (`[]`), the agent runs in text-only mode and reweights scores accordingly |
| `style_summary` | `str` | No | Narrative description — words longer than 3 chars are added to the keyword pool. Leave empty in eval to avoid diluting scores with generic terms |

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
| `image_embedding` | `list[float]` | No | Pre-computed product image embedding. If empty, image similarity is skipped for this product |
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
    "overall_reject_threshold": 0.35,      # default — lower for text-only mode
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

Products that failed critic evaluation, with a human-readable reason. This list can be summarized into `critic_feedback` for a procurement retry pass.

```python
[
    {
        "product_id": "p_14",
        "reason": "Rejected: overall score too low (0.20 < 0.35)."
    },
    {
        "product_id": "p_27",
        "reason": "Rejected: must-avoid constraint violated ('synthetic' found in product)."
    },
    ...
]
```

### `ranker_critic_output`
**Type:** `dict`

Full structured output including score breakdowns for every accepted product. Useful for debugging and building `critic_feedback`.

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

---

## Scoring logic

### Three-dimensional scoring

Each product is scored on three axes:

| Dimension | Default weight | Method |
|-----------|---------------|--------|
| Image similarity | 50% | Cosine similarity between `board_embedding` and `product["image_embedding"]` |
| Text similarity | 30% | `keyword_overlap_score(style_keywords + style_summary_words, product_text)` — words from `style_summary` longer than 3 chars are added to the keyword pool; `style_elements` are NOT included here |
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

A product with no keyword overlap and no avoid violations has a semantic score of exactly `0.50`. In text-only mode this gives a final score of `0.60×0 + 0.40×0.5 = 0.20` — below the default rejection threshold of `0.35`. This is the source of the ~2,800 floor rejections in the offline eval.

---

## Critic rejection rules

Applied in order before the final score is calculated:

1. **Must-avoid auto-reject** — if any term from `constraints.must_avoid` appears anywhere in the product text, the product is immediately rejected with no score computed.
2. **Image similarity floor** — if image embeddings are present and `image_similarity < 0.25`, the product is rejected. Not applied in text-only mode.
3. **Avoid-list multi-violation** — if 2 or more avoid-list terms (colors or materials) are found, the product is rejected regardless of score.
4. **Overall score threshold** — if the weighted final score is below `0.35`, the product is rejected.

---

## Known limitations

**Image embeddings are not computed anywhere in the current pipeline.** Procurement returns `image_url` strings; the ranker expects pre-computed embedding vectors. The 50% image similarity weight is effectively dead weight — the ranker always runs in text-only mode. To activate image scoring, embeddings would need to be computed (e.g. with CLIP or SigLIP) at procurement time or as a separate preprocessing step.

**Rejection threshold is calibrated for image+text mode.** At `0.35`, the threshold is appropriate when image similarity contributes 50% of the score. In text-only mode, the same threshold rejects any product where style keywords don't appear in the product name — about 79% of all candidates in the offline eval. Lowering to `overall_reject_threshold * 0.7` when `has_image_scores is False` is the recommended fix.

---

## Setup

No external API keys required. The ranker is entirely local — no model is loaded, no network calls are made. All scoring is implemented in pure Python using utilities from `src/utils/helper.py`.

```bash
pip install -r requirements.txt
```

No additional dependencies beyond what's already in `requirements.txt`.

---

## Testing the agent in isolation

```python
# Direct call — no state overhead needed for unit testing

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
```

Run with:
```bash
cd VisionCart
python tests/test_eval.py --gt-only --style dark_academia
```

The `--gt-only` mode in `test_eval.py` is the fastest way to verify ranker behavior — it runs the ranker against only the 4 known-correct products per query, so any recall below 1.0 is a pure threshold or scoring problem.

---

## Connecting to the pipeline

The ranker sits between procurement and output in `main.py`:

```python
# main.py (simplified)
ranker_results = ranker_critic.run(state)   # reads style_profile + candidate_products
state.update(ranker_results)                # writes ranked_products, rejected_products
output_results = output.run(state)          # reads ranked_products
```

**Critical:** `state["style_profile"]` must be the structured dict described above — not the narrative string that procurement returns. Currently `main.py` sets `state["style_profile"] = procurement_data["style_profile"]`, which is a plain string. `_ensure_dict()` in the ranker returns `{}` for a plain string, so the ranker has no style signal in production. This is tracked as fix 1 in `eval_tasks_2026-04-20.md`.


> Cross-agent coordination items for this agent are tracked in [`docs/eval_tasks_2026-04-20.md`](eval_tasks_2026-04-20.md) under **Cross-Agent Coordination Tasks**.
