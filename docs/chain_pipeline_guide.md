# VisionCart — Sequential Chain Pipeline Guide
`src/main-LC.py`

---

## What this file does

`main-LC.py` is the sequential entry point for the VisionCart pipeline. It is not an agent — it owns no model and does no scoring. Its job is to:

1. **Crawl a Pinterest board URL** — downloads images to a local `data/` folder via `utils/pinterest_crawler.py`
2. **Initialize the shared state dict** — the single object every agent reads from and writes to
3. **Call each agent in sequence** — stylist → procurement → ranker/critic → output
4. **Stitch the state together between agents** — each agent returns a partial dict; `main-LC.py` merges it back before calling the next
5. **Print the final result** — style identified, number of products that passed the critic, and the output agent's narrative summary

> For the LangGraph version with automatic retry logic, see [`graph_pipeline_guide.md`](graph_pipeline_guide.md) (`src/main-LG.py`).

---

## Pipeline flow

```
Pinterest board URL (CLI argument)
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
        │  returns: JSON string → procurement_products, procurement_queries, style_profile
        │
_style_profile_from_stylist(stylist_output)
        │  builds ranker-compatible style_profile dict
        ▼
ranker_critic.run(state)
        │  reads:  style_profile, candidate_products
        │  writes: ranked_products, rejected_products, ranker_critic_output, critic_feedback
        ▼
output.run(state)
        │  reads:  style_profile, ranked_products
        │  writes: output_text, output_products
        ▼
print final summary
```

---

## Inputs

### `url` (required, positional CLI argument)

The public Pinterest board URL. Must be a valid public board — private boards and search URLs are not supported by the crawler.

```bash
python src/main-LC.py https://www.pinterest.com/username/board-name/
```

### `--max-images` (optional, default `5`)

How many images to download from the board and pass to the stylist. The stylist internally caps at 5 regardless, so increasing this beyond 5 has no effect until that limit is raised in `stylist.py`.

```bash
python src/main-LC.py https://www.pinterest.com/username/board-name/ --max-images 3
```

---

## State schema

The shared state is initialized as an `AgentState` TypedDict (defined in `src/graph/state.py`) and passed through every agent call.

| Key | Set by | Type | Description |
|-----|--------|------|-------------|
| `vision_board_paths` | `main-LC.py` | `list[str]` | Local paths to downloaded board images |
| `num_products` | `main-LC.py` | `int` | Target product count (hardcoded to `5`) |
| `iterations` | `main-LC.py` | `int` | Initialized to `0`; not incremented in this chain variant |
| `stylist_output` | `stylist` | `dict` | Style persona: `style_profile`, `aesthetic`, `colors`, `materials`, `products` |
| `procurement_queries` | `main-LC.py` | `list[str]` | LLM-generated Google Shopping queries |
| `candidate_products` | `main-LC.py` | `list[dict]` | Raw product pool from SerpAPI |
| `style_profile` | `main-LC.py` | `dict` | Ranker-compatible style dict built by `_style_profile_from_stylist()` |
| `ranked_products` | `ranker/critic` | `list[dict]` | Accepted products sorted by score |
| `rejected_products` | `ranker/critic` | `list[dict]` | Rejected products with reasons |
| `ranker_critic_output` | `ranker/critic` | `dict` | Full score breakdowns for all evaluated products |
| `critic_feedback` | `ranker/critic` | `str \| None` | Feedback for a hypothetical retry; `None` if results are satisfactory |
| `output_text` | `output` | `str` | Human-readable shopping summary |
| `output_products` | `output` | `list[dict]` | Top 5 products structured for UI card rendering |

---

## `_style_profile_from_stylist()` — the style profile bridge

Procurement's `run()` returns `style_profile` as a plain narrative string. The ranker needs a **structured dict** with `style_keywords`, `color_palette`, and `materials`. This helper builds that dict from `stylist_output`, which already contains the structured fields:

```python
# main-LC.py lines 11–39
def _style_profile_from_stylist(stylist_output: Dict[str, Any]) -> Dict[str, Any]:
    aesthetic = stylist_output.get("aesthetic") or []
    colors    = stylist_output.get("colors")    or []
    materials = stylist_output.get("materials") or []
    narrative = stylist_output.get("style_profile", "")
    return {
        "board_id":       "",
        "style_summary":  narrative,
        "style_keywords": list(dict.fromkeys(aesthetic + colors)),
        "style_elements": aesthetic,
        "color_palette": {
            "dominant": colors,
            "accent":   [],
            "avoid":    [],
        },
        "materials": {
            "preferred": materials,
            "avoid":     [],
        },
        "constraints":    {"must_avoid": []},
        "board_embedding": [],
    }
```

Called at line 101, after procurement runs:

```python
state["style_profile"] = _style_profile_from_stylist(state["stylist_output"])
```

---

## Running the pipeline

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| GPU (≥14 GB VRAM) | Required for stylist (Qwen2.5-VL-7B). Google Colab T4/A100 works. |
| `SERPAPI_API_KEY` | Required for product search |
| `HF_TOKEN` | Required only if not using Ollama for query generation |
| `OLLAMA_MODEL` | Optional; set to use local Ollama (e.g. `qwen2.5:7b`) for both query gen and output |
| Python dependencies | `pip install -r requirements.txt` |

`.env` at repo root:

```bash
SERPAPI_API_KEY=your_serpapi_key_here
HF_TOKEN=your_huggingface_token_here   # omit if using Ollama
OLLAMA_MODEL=qwen2.5:7b               # omit if using HuggingFace
```

### Run from terminal

```bash
cd VisionCart
python src/main-LC.py https://www.pinterest.com/aesthetics/spring-wallpapers/
python src/main-LC.py https://www.pinterest.com/aesthetics/spring-wallpapers/ --max-images 1
```

### Run from Google Colab

Use `src/agents/stylistGC.ipynb`. Cell 7 runs the full pipeline via:

```python
%run src/main-LC.py https://www.pinterest.com/aesthetics/spring-wallpapers/
```

---

## What gets printed

```
crawling pinterest board: https://...
images downloaded to: data/2026-04-20_12-00-00/

stylist agent analysis
--- RAW MODEL OUTPUT ---
{ "style_profile": "...", ... }
------------------------

procurement agent search
[procurement] Calling local ollama (model: qwen2.5:7b)...
[procurement] Fetching products for 5 queries (up to 20 results each)...
[procurement] Done. Total candidate pool: 87 products.

ranker/critic evaluation
[ranker_critic] Candidates received: 87
[ranker_critic] Results: 12 accepted, 75 rejected

output generation
[output] ranked_products: 12  style_profile: 210 chars

complete analysis
Style Identified: {'style_summary': 'Warm scholarly atmosphere...', ...}
Results: 12 products passed the critic.
------------------------------
Your board reads as dark academia with an emphasis on...
```

---

## Setup

```bash
pip install -r requirements.txt
```

Verify GPU (required for stylist):

```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # e.g. "Tesla T4"
```

---

## Testing in isolation

```bash
python src/main-LC.py https://www.pinterest.com/aesthetics/spring-wallpapers/ --max-images 1
```

To bypass the Pinterest crawler and use local images:

```python
import json
import src.agents.stylist as stylist
import src.agents.procurement as procurement
import src.agents.ranker_critic as ranker_critic
import src.agents.output as output
from src.main_LC import _style_profile_from_stylist

state = {
    "vision_board_paths": [
        "dataset/images/dark_academia/01_loft_womens_herringbone_relaxed_blazer.jpg",
        "dataset/images/dark_academia/02_brown_leather_satchel_bag.jpg",
    ],
    "num_products": 5,
    "iterations": 0,
    "stylist_output": {},
    "candidate_products": [],
    "ranked_products": [],
    "output_text": ""
}

state.update(stylist.run(state))
proc_data = json.loads(procurement.run(state))
state["candidate_products"] = proc_data["procurement_products"]
state["procurement_queries"] = proc_data["procurement_queries"]
state["style_profile"] = _style_profile_from_stylist(state["stylist_output"])
state.update(ranker_critic.run(state))
state.update(output.run(state))
print(state["output_text"])
```
