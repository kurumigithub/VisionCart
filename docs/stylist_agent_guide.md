# VisionCart — Stylist Agent Guide
`src/agents/stylist.py`

---

## What this agent does

The stylist agent is the first step in the VisionCart pipeline. It receives a set of local image paths from a Pinterest board and converts them into a structured style persona — a machine-readable description of the user's aesthetic that every downstream agent uses.

1. **Reads image paths from state** — loads up to 5 images as PIL objects. Images beyond 5 are dropped to manage GPU memory and token limits.
2. **Constructs a multimodal prompt** — attaches all images alongside a structured JSON schema instruction and sends them together to **Qwen2.5-VL-7B-Instruct**, a vision-language model running locally.
3. **Parses the model response** — looks for a JSON block inside markdown fences first, then falls back to the last `{…}` pair in the output string. If both fail, returns an empty shell with a warning rather than crashing.
4. **Writes the style persona to state** — the output dict is the contract that procurement and the ranker build on. The schema must be exact.

---

## Input: State keys

### `vision_board_paths` (required)

A list of local file paths to images downloaded from the user's Pinterest board. The crawler in `utils/pinterest_crawler.py` produces this list and writes it into state before the stylist runs.

```python
state["vision_board_paths"] = [
    "data/2026-04-20_12-00-00/image_01.jpg",
    "data/2026-04-20_12-00-00/image_02.jpg",
    "data/2026-04-20_12-00-00/image_03.jpg",
]
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `vision_board_paths` | `list[str]` | Yes | Local file paths to board images. Only the first 5 are used. Must be `.jpg`, `.jpeg`, or `.png`. |

---

## Output: Style Persona

The agent writes one key to state:

### `stylist_output`
**Type:** `dict`

A structured style persona extracted from the vision board. This is the primary contract between the stylist and every agent downstream.

```json
{
  "style_profile": "Warm scholarly atmosphere with rich earth tones and layered textures — think autumn library, vintage leather, and candlelit reading nooks.",
  "products": ["blazer", "leather bag", "wool scarf", "oxford shoes"],
  "aesthetic": ["dark academia", "vintage prep", "scholarly"],
  "colors": ["burgundy", "forest green", "camel", "cream"],
  "materials": ["wool", "leather", "tweed", "velvet"]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `style_profile` | `str` | Narrative prose description of the overall vibe — used by procurement as the primary LLM prompt input and by output as the user-facing style summary |
| `products` | `list[str]` | Specific item types the stylist identified in the images (e.g. `"planter"`, `"blazer"`) — informational for procurement, not directly used in query generation |
| `aesthetic` | `list[str]` | Mood and vibe labels — used by procurement for query context and by the ranker's semantic scoring |
| `colors` | `list[str]` | 3–5 specific color names found in the images — used by procurement and the ranker's color palette matching |
| `materials` | `list[str]` | 3–5 material types identified — used by procurement and the ranker's material preference rules |

---

## The vision model

The stylist uses **Qwen/Qwen2.5-VL-7B-Instruct**, a multimodal vision-language model that processes both images and text in a single forward pass. It runs locally on GPU via HuggingFace Transformers.

The model is loaded at module import time — not lazily inside `run()`. This means the first import of `stylist.py` downloads and loads a 7B-parameter model into GPU memory. On a T4 (16 GB VRAM), this takes ~2 minutes on first run and is cached afterward.

The prompt enforces the exact JSON schema the downstream agents expect:

```
Analyze these images and create a 'Style Persona'.
Return ONLY a JSON object with these exact keys:
{
  "style_profile": "...",
  "products": [...],
  "aesthetic": [...],
  "colors": [...],
  "materials": [...]
}
```

---

## JSON parsing and failure modes

The model sometimes wraps its JSON in markdown fences (` ```json ... ``` `), sometimes outputs raw JSON, and occasionally produces valid JSON buried inside explanatory text. The parser handles all three cases:

1. **Primary path** — looks for ` ```json { ... } ``` ` with a regex
2. **Fallback** — finds the last `{ ... }` pair in the full output string (avoids picking up JSON from the model's own chain-of-thought preamble)
3. **Failure** — if both fail, logs the error and returns an empty shell:

```python
{
    "style_profile": "Unknown",
    "products": [],
    "aesthetic": [],
    "colors": [],
    "materials": []
}
```

The empty shell does not crash the pipeline but will cause procurement to generate generic queries and the ranker to have no style signal. Check the raw model output printed to stdout when debugging — it is always logged regardless of parse outcome.

---

## Setup

### Hardware requirements

The stylist requires a GPU with at least **14 GB VRAM** to run Qwen2.5-VL-7B at full precision (`torch_dtype="auto"`). The Google Colab setup in `src/agents/stylistGC.ipynb` uses a T4 (16 GB) or A100.

Running on CPU is technically possible but impractically slow (~10–20 minutes per image set).

### Install dependencies

```bash
pip install -r requirements.txt
```

Key packages for this agent:

```
torch                          # GPU inference
transformers>=4.43.0           # Qwen2.5-VL support
accelerate                     # device_map="auto" support
qwen-vl-utils                  # image preprocessing for Qwen-VL
Pillow                         # PIL image loading
python-dotenv>=1.0.0           # loads .env
```

### API keys

```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — read access is sufficient. The token is used to authenticate the model download. After the first run the model is cached locally in `~/.cache/huggingface/`.

---

## Testing the agent in isolation

```python
# tests/test_stylist_agent.py

import sys
sys.path.insert(0, "src")

from agents.stylist import run

mock_state = {
    "vision_board_paths": [
        "dataset/images/dark_academia/01_loft_womens_herringbone_relaxed_blazer.jpg",
        "dataset/images/dark_academia/02_brown_leather_satchel_bag.jpg",
    ]
}

result = run(mock_state)
output = result["stylist_output"]

print("style_profile:", output["style_profile"])
print("aesthetic:", output["aesthetic"])
print("colors:", output["colors"])
print("materials:", output["materials"])

# Validate schema
assert isinstance(output["style_profile"], str) and output["style_profile"] != "Unknown"
assert isinstance(output["aesthetic"], list) and len(output["aesthetic"]) > 0
assert isinstance(output["colors"], list) and len(output["colors"]) > 0
assert isinstance(output["materials"], list) and len(output["materials"]) > 0
print("All assertions passed.")
```

Run it with:
```bash
cd VisionCart
python tests/test_stylist_agent.py
```

The `dataset/images/` directory already contains 120 product images organized by style — use these as test inputs without needing to crawl Pinterest.

---

## Running on Google Colab

The notebook at `src/agents/stylistGC.ipynb` is the recommended way to run the stylist agent. It handles GPU setup, HuggingFace authentication, and Google Drive mounting automatically.

```
Cell 1: Enter HF_TOKEN and SERPAPI_API_KEY via getpass (not hardcoded)
Cell 2: Verify GPU (expects Tesla T4 or better)
Cell 3: Authenticate with HuggingFace
Cell 4: Mount Google Drive and navigate to project directory
Cell 5: Install GPU-optimized dependencies
Cell 6: Add src/ to sys.path for local imports
Cell 7: Run full pipeline via %run src/main.py
```

---

## Connecting to the pipeline

The stylist is the entry point of the LangGraph graph. It reads `vision_board_paths` (written by the Pinterest crawler in `main.py`) and writes `stylist_output` for procurement:

```python
# main.py (simplified)
state["vision_board_paths"] = image_paths   # from crawler
state.update(stylist.run(state))            # writes stylist_output
procurement_raw = procurement.run(state)    # reads stylist_output
```

The `stylist_output` dict is also the source of truth for building the ranker's `style_profile` — see the known bug in `eval_tasks_2026-04-20.md` where `main.py` currently passes the wrong type to the ranker.


> Cross-agent coordination items for this agent are tracked in [`docs/eval_tasks_2026-04-20.md`](eval_tasks_2026-04-20.md) under **Cross-Agent Coordination Tasks**.
