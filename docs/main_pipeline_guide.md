# VisionCart — Pipeline Orchestrator Guide
`src/main.py`

---

## What this file does

`main.py` is the entry point and orchestrator for the full VisionCart pipeline. It is not an agent — it owns no model and does no scoring. Its job is to:

1. **Crawl a Pinterest board URL** — downloads images to a local `data/` folder via `utils/pinterest_crawler.py`
2. **Initialize the shared state dict** — the single object every agent reads from and writes to
3. **Call each agent in sequence** — stylist → procurement → ranker/critic → output
4. **Stitch the state together between agents** — each agent returns a partial dict; `main.py` merges it back before calling the next
5. **Print the final result** — style identified, number of products that passed the critic, and the output agent's narrative summary

---

## Pipeline flow

```
Pinterest board URL
        │
        ▼
pinterest_crawler.crawl_pinterest()
        │  downloads images to data/YYYY-MM-DD_HH-MM-SS/
        ▼
state["vision_board_paths"] = [local image paths]
        │
        ▼
stylist.run(state)
        │  writes: stylist_output
        ▼
procurement.run(state)
        │  reads:  stylist_output
        │  writes: candidate_products, procurement_queries, style_profile (string)
        ▼
ranker_critic.run(state)
        │  reads:  style_profile, candidate_products
        │  writes: ranked_products, rejected_products, ranker_critic_output
        ▼
output.run(state)
        │  reads:  style_profile, ranked_products
        │  writes: output_text, output_products
        ▼
print final summary
```

---

## Inputs

### `board_url` (required)

The public Pinterest board URL passed to `run_vision_cart()`. Must be a valid public board — private boards and search URLs are not supported by the crawler.

```python
run_vision_cart("https://www.pinterest.com/username/board-name/")
```

### `max_images` (optional, default `5`)

How many images to download from the board and pass to the stylist. The stylist internally caps at 5 regardless, so increasing this beyond 5 has no effect until that limit is raised in `stylist.py`.

---

## State schema

The shared state is initialized as an `AgentState` TypedDict (defined in `src/graph/state.py`) and passed through every agent call. After the full pipeline runs, the state contains:

| Key | Set by | Type | Description |
|-----|--------|------|-------------|
| `vision_board_paths` | `main.py` | `list[str]` | Local paths to downloaded board images |
| `num_products` | `main.py` | `int` | Target product count (currently hardcoded to `5`) |
| `stylist_output` | `stylist` | `dict` | Style persona: `style_profile`, `aesthetic`, `colors`, `materials`, `products` |
| `procurement_queries` | `procurement` | `list[str]` | LLM-generated Google Shopping queries |
| `candidate_products` | `main.py` | `list[dict]` | Raw product pool from SerpAPI (parsed from procurement's JSON string) |
| `style_profile` | `main.py` | `str` | Narrative style string from procurement — **currently a string, not the ranker-compatible dict** |
| `ranked_products` | `ranker/critic` | `list[dict]` | Accepted products sorted by score |
| `rejected_products` | `ranker/critic` | `list[dict]` | Rejected products with reasons |
| `output_text` | `output` | `str` | Human-readable shopping summary |
| `output_products` | `output` | `list[dict]` | Top 5 products for UI rendering |

---

## Known bug: `style_profile` type mismatch

This is the most critical issue in the current pipeline. The ranker needs `style_profile` to be a **structured dict** with `style_keywords`, `color_palette`, and `materials`. But `main.py` currently sets it to a **plain narrative string** from procurement:

```python
# main.py lines 64–68 — current (broken) behavior
procurement_raw = procurement.run(state)
procurement_data = json.loads(procurement_raw)
state["candidate_products"] = procurement_data["procurement_products"]
state["procurement_queries"] = procurement_data["procurement_queries"]
state["style_profile"] = procurement_data["style_profile"]   # ← plain string, e.g. "Scholarly, vintage-inspired..."
```

The ranker calls `_ensure_dict()` on this value. A plain narrative string is not valid JSON, so `_ensure_dict()` silently returns `{}`. The ranker then scores every product with an empty style profile — no keywords, no palette, no material preferences.

**The fix** is to build the ranker-compatible dict from `stylist_output` before calling the ranker:

```python
# Proposed fix — replace lines 64–68 with:
procurement_raw = procurement.run(state)
procurement_data = json.loads(procurement_raw)
state["candidate_products"] = procurement_data["procurement_products"]
state["procurement_queries"] = procurement_data["procurement_queries"]

stylist_out = state["stylist_output"]
state["style_profile"] = {
    "style_summary":  procurement_data["style_profile"],
    "style_keywords": stylist_out.get("aesthetic", []) + stylist_out.get("materials", []),
    "style_elements": [],
    "color_palette": {
        "dominant": stylist_out.get("colors", []),
        "accent":   [],
        "avoid":    [],
    },
    "materials": {
        "preferred": stylist_out.get("materials", []),
        "avoid":     [],
    },
    "constraints":    {"must_avoid": []},
    "board_embedding": [],
}
```

This is tracked as fix 1 in `eval_tasks_2026-04-20.md`.

---

## Running the pipeline

### Prerequisites

| Requirement | Where to get it |
|-------------|----------------|
| GPU (≥14 GB VRAM) | Google Colab T4/A100, or local CUDA GPU |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — read access |
| `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com) |
| Python dependencies | `pip install -r requirements.txt` |

Set keys in a `.env` file at the repo root:

```bash
# .env
HF_TOKEN=your_huggingface_token_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### Run from terminal

```bash
cd VisionCart
python src/main.py
```

The board URL is currently hardcoded at line 102 for testing:

```python
test_url = "https://www.pinterest.com/aesthetics/spring-wallpapers/"
run_vision_cart(test_url, max_images=1)
```

To use a different board, edit that line or uncomment the `argparse` block above it (lines 87–98) and pass the URL as a CLI argument:

```bash
python src/main.py https://www.pinterest.com/username/board-name/
```

### Run from Google Colab

Use `src/agents/stylistGC.ipynb`. Cell 7 runs the full pipeline via:

```python
%run src/main.py
```

---

## What gets printed

A successful run prints at each stage:

```
crawling pinterest board: https://...
images downloaded to: data/2026-04-20_12-00-00/

stylist agent analysis
--- RAW MODEL OUTPUT ---
{ "style_profile": "...", ... }
------------------------

procurement agent search
[procurement] Calling HuggingFace Inference API (model: Qwen/Qwen2.5-7B-Instruct)...
[procurement] Fetching products for 5 queries (up to 20 results each)...
[procurement] Done. Total candidate pool: 87 products.

ranker/critic evaluation
[ranker_critic] Candidates received: 87
[ranker_critic] Results: 12 accepted, 75 rejected

output generation
[output] ranked_products count: 12

complete analysis
Style Identified: Warm scholarly atmosphere with rich earth tones...
Results: 12 products passed the critic.
------------------------------
Your board reads as dark academia with an emphasis on...
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify environment

```python
import torch
print(torch.cuda.is_available())       # must be True for stylist to run
print(torch.cuda.get_device_name(0))   # e.g. "Tesla T4"
```

---

## Testing in isolation

`main.py` can be tested end-to-end by pointing it at a board URL with `max_images=1` to minimize GPU time and API costs:

```python
from src.main import run_vision_cart

state = run_vision_cart(
    "https://www.pinterest.com/aesthetics/spring-wallpapers/",
    max_images=1
)

# Verify state has expected keys after full run
assert "stylist_output" in state
assert "candidate_products" in state
assert "ranked_products" in state
assert "output_text" in state
print("Pipeline completed successfully.")
print(f"Products accepted: {len(state['ranked_products'])}")
print(state["output_text"])
```

To test without hitting Pinterest (e.g. using local dataset images):

```python
from src.main import run_vision_cart
import src.agents.stylist as stylist
import src.agents.procurement as procurement
import src.agents.ranker_critic as ranker_critic
import src.agents.output as output
import json

state = {
    "vision_board_paths": [
        "dataset/images/dark_academia/01_loft_womens_herringbone_relaxed_blazer.jpg",
        "dataset/images/dark_academia/02_brown_leather_satchel_bag.jpg",
    ],
    "num_products": 5,
    "stylist_output": {},
    "candidate_products": [],
    "ranked_products": [],
    "output_text": ""
}

state.update(stylist.run(state))
proc_data = json.loads(procurement.run(state))
state["candidate_products"] = proc_data["procurement_products"]
state["procurement_queries"] = proc_data["procurement_queries"]
state["style_profile"] = proc_data["style_profile"]   # note: broken until fix 1 is applied
state.update(ranker_critic.run(state))
state.update(output.run(state))
print(state["output_text"])
```


> Cross-agent coordination items for this file are tracked in [`docs/eval_tasks_2026-04-20.md`](eval_tasks_2026-04-20.md) under **Cross-Agent Coordination Tasks**.
