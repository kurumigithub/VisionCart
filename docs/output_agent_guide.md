# VisionCart — Output Agent Guide
`src/agents/output.py`

---

## What this agent does

The output agent is the last step in the pipeline. It receives the ranked product list and style profile from the ranker/critic, calls a local Ollama LLM to generate a warm, human-readable shopping summary, and packages the top 5 products into a structured list for UI rendering.

```
state flows in → output.py reads it → calls Ollama → writes output_text + output_products → state flows out
```

---

## Input: State keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `style_profile` | `str \| dict` | Yes | The user's aesthetic description. If missing or empty, returns an early error message. |
| `ranked_products` | `list[dict]` | Yes | Accepted products from the ranker, sorted by score. |
| `critic_notes` | `str` | No | Optional rejection notes from the critic — included in the LLM prompt if present. |

If `style_profile` is empty, the agent returns immediately with a "couldn't determine your style profile" message and an empty `output_products` list. Same for an empty `ranked_products` list.

---

## Output: State keys

### `output_text`
**Type:** `str`

Narrative shopping summary from the Ollama LLM. Formatted as:
1. A 1–2 sentence interpretation of the style profile
2. Top 3–5 products with a brief explanation of why each fits
3. An optional note if results were sparse or any products were filtered

### `output_products`
**Type:** `list[dict]`

Top 5 products from `ranked_products`, structured for UI card rendering. Each item has:

```python
{
    "name": "Loft Women's Herringbone Relaxed Blazer",
    "price": 89.99,
    "url": "https://...",
    "image_url": "https://...",
    "score": 0.724,
    "tags": ["blazer", "wool", "herringbone", "brown"]
}
```

---

## LLM backend: Ollama

The output agent uses a **local Ollama instance** for text generation — not the Anthropic Claude API. The model and host are configurable via environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model to use for generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server base URL |

The agent calls `POST /api/chat` on the Ollama server with `stream: false` and `think: false`. If the model returns residual `<think>...</think>` blocks (some models ignore `think: false`), they are stripped from the response.

---

## Key functions

### `_call_ollama(system, user) -> str`

Sends a system + user message to the local Ollama server and returns the model's text response. Strips any `<think>` blocks. Raises `requests.HTTPError` if the server returns a non-2xx status.

### `format_products_for_prompt(ranked_products) -> str`

Converts the top 5 ranked products into a readable text block for the LLM prompt:

```
1. Loft Women's Herringbone Relaxed Blazer
   Score: 0.7241
   Tags: blazer, wool, herringbone, brown
   Price: 89.99
   URL: https://...
```

### `build_output_products(ranked_products) -> list`

Builds the structured `output_products` list for UI rendering. Takes the top 5 from `ranked_products` and normalizes field names, rounding score to 3 decimal places.

---

## System prompt

```
You are a personal shopping assistant for a visual-first retail app.
Your job is to translate AI-ranked product results into a warm,
concise, human-readable summary that explains why the results
match the user's aesthetic and helps them make a decision.

Format your response as:
1. A 1-2 sentence interpretation of the user's style profile
2. Top 3-5 products with a brief explanation of why each fits
3. An optional note if results were sparse or any products were filtered

Keep the tone conversational and specific — reference actual
aesthetic details like color, texture, silhouette, or vibe.
Do not use filler phrases like "Great news!" or "I found some products."
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

The output agent only needs `requests` (for Ollama HTTP calls) — no Anthropic SDK required.

### Run Ollama locally

```bash
ollama pull qwen2.5:7b     # first-time download
ollama serve               # starts server at http://localhost:11434
```

### Environment variables

```bash
# .env (optional — defaults work if Ollama is running locally)
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

---

## Testing the agent in isolation

```python
# tests/test_output_agent.py

import sys
sys.path.insert(0, "src")

from agents.output import run

mock_state = {
    "style_profile": "Warm earth tones, relaxed Japanese streetwear silhouettes, natural textures like linen and washed denim, minimal branding.",
    "ranked_products": [
        {
            "name": "Uniqlo Wide Linen Trousers",
            "score": 0.91,
            "tags": ["linen", "wide-leg", "neutral", "minimalist"],
            "url": "https://uniqlo.com/...",
            "image_url": "",
            "price": 49.90
        },
        {
            "name": "Mango Washed Denim Overshirt",
            "score": 0.84,
            "tags": ["denim", "oversized", "earth tone", "texture"],
            "url": "https://mango.com/...",
            "image_url": "",
            "price": 69.99
        }
    ],
    "critic_notes": "Filtered 2 products that matched color but had prominent logo branding."
}

result = run(mock_state)
print(result["output_text"])
print(f"\noutput_products count: {len(result['output_products'])}")
```

Run it with:
```bash
cd VisionCart
python tests/test_output_agent.py
```

---

## Connecting to the pipeline

The output agent is the final node in both orchestrators:

```python
# main-LC.py (simplified)
output_results = output.run(state)   # reads ranked_products, style_profile
state.update(output_results)         # writes output_text, output_products
print(state["output_text"])
```

In the LangGraph version (`main-LG.py`), the `output_node` wrapper additionally checks the best product score and prepends `"LOW MATCH CONFIDENCE\n"` to `output_text` if `max_score <= 0.6` (indicating the retry loop exhausted its attempts without finding a strong match).

---

## What good output looks like

> Your board reads as relaxed minimalism with an earth-tone palette — think undyed linen, washed indigo, and warm taupe, with a preference for loose, unconstructed silhouettes.
>
> **1. Uniqlo Wide Linen Trousers** — The texture and drape are a near-exact match for the linen pieces in your board. The taupe colorway lines up with 3 of your 7 reference images.
>
> **2. Mango Washed Denim Overshirt** — The worn-in finish and boxy cut fit your streetwear references. The critic flagged a similar jacket with visible branding, so this lower-logo option was ranked higher.
>
> 2 additional results were filtered for not meeting your vibe threshold — they matched keyword tags but had a sharper, more polished aesthetic than your board suggests.
