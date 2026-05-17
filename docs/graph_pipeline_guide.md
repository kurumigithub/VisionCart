# VisionCart — LangGraph Pipeline Guide
`src/main-LG.py`

---

## What this file does

`main-LG.py` is the LangGraph-native orchestration variant of the VisionCart pipeline. Unlike the sequential chain in `main-LC.py`, this version defines a **stateful graph** with a conditional retry loop: if the ranker/critic finds no strong product matches (max score ≤ 0.6), procurement is called again with the critic's feedback — up to 3 total attempts.

> For the simpler sequential version without retry logic, see [`main_pipeline_chain_guide.md`](chain_pipeline_guide.md) (`src/main-LC.py`).

---

## Graph structure

```
          ┌─────────┐
          │ stylist │  (entry point)
          └────┬────┘
               │
          ┌────▼──────────┐
     ┌───►│  procurement  │◄────────────────┐
     │    └────┬──────────┘                 │
     │         │                            │
     │    ┌────▼────┐   should_retry()      │
     │    │  critic │ ──────────────────────┘
     │    └────┬────┘  "retry" (max_score ≤ 0.6 AND iterations < 3)
     │         │
     │         │ "continue" (max_score > 0.6 OR iterations ≥ 3)
     │    ┌────▼────┐
     └────│ output  │
          └────┬────┘
               │
              END
```

---

## Nodes

### `stylist_node`
Calls `stylist.run(state)`. Writes `stylist_output`. Runs exactly once — the graph entry point.

### `procurement_node`
Calls `procurement.run(state)`, parses the returned JSON string, and calls `_style_profile_from_stylist()` to build the ranker-compatible `style_profile` dict. Also increments `state["iterations"]` by 1 each time it runs.

On retry passes, `state["critic_feedback"]` (set by the critic node on the previous pass) is already in state and gets picked up automatically by `procurement.run()`.

### `critic_node`
Calls `ranker_critic.run(state)`. Writes `ranked_products`, `rejected_products`, `ranker_critic_output`, and `critic_feedback`.

### `output_node`
Calls `output.run(state)`. Writes `output_text` and `output_products`. If the best product score is ≤ 0.6 (meaning max retries were exhausted without finding a strong match), prepends `"LOW MATCH CONFIDENCE\n"` to `output_text` as a warning.

---

## Routing logic

After the critic node runs, `should_retry()` decides whether to loop back to procurement or proceed to output:

```python
def should_retry(state) -> Literal["retry", "continue"]:
    ranked_products = state.get("ranked_products", [])
    iterations = state.get("iterations", 0)
    max_score = max([p["score"] for p in ranked_products]) if ranked_products else 0

    if max_score > 0.6:
        return "continue"     # strong match found — proceed to output
    if iterations < 3:
        return "retry"        # try procurement again with critic feedback
    return "continue"         # max retries reached — use best available
```

| Condition | Route | Explanation |
|-----------|-------|-------------|
| `max_score > 0.6` | `continue` | At least one product strongly matches the aesthetic |
| `max_score ≤ 0.6` and `iterations < 3` | `retry` | Retry procurement with `critic_feedback` |
| `max_score ≤ 0.6` and `iterations ≥ 3` | `continue` | Exhausted retries; output warns of low confidence |

The `iterations` counter is incremented by `procurement_node`, so after the first run it equals 1. After the second it equals 2. The condition `iterations < 3` allows up to 2 retries (3 total procurement calls).

---

## Retry mechanism

On a retry pass, `critic_feedback` from the previous critic run is already in state. `procurement.run()` reads it via `state.get("critic_feedback")` and forwards it to the LLM query builder, which generates different query angles for the failing categories.

The `style_profile` dict is rebuilt from `stylist_output` at the start of every `procurement_node` call, so the ranker always gets a fresh structured profile regardless of how many retries have occurred.

---

## State schema

The graph uses `AgentState` from `src/graph/state.py`. The LangGraph version initializes all keys upfront:

| Key | Initial value | Set/updated by |
|-----|--------------|----------------|
| `vision_board_paths` | crawled image paths | `run_vision_cart()` |
| `num_products` | `5` | `run_vision_cart()` |
| `iterations` | `0` | `procurement_node` (incremented each pass) |
| `stylist_output` | `{}` | `stylist_node` |
| `procurement_queries` | `[]` | `procurement_node` |
| `candidate_products` | `[]` | `procurement_node` |
| `style_profile` | `""` | `procurement_node` (via `_style_profile_from_stylist`) |
| `ranked_products` | `[]` | `critic_node` |
| `rejected_products` | `[]` | `critic_node` |
| `critic_feedback` | not set initially | `critic_node` (on each pass) |
| `output_text` | `""` | `output_node` |
| `output_products` | `[]` | `output_node` |

---

## `_style_profile_from_stylist()` — the style profile bridge

Defined identically to the chain version. Procurement's `run()` returns `style_profile` as a plain narrative string; the ranker needs a structured dict. This helper builds it from `stylist_output`:

```python
state["style_profile"] = _style_profile_from_stylist(state["stylist_output"])
```

See [`chain_pipeline_guide.md`](chain_pipeline_guide.md) for the full implementation.

---

## Running the pipeline

### Prerequisites

Same as the chain version — see [Prerequisites](chain_pipeline_guide.md#prerequisites) in `chain_pipeline_guide.md`.

### Run

The LangGraph version currently has the board URL hardcoded at the bottom of the file:

```python
# main-LG.py line 176
test_url = "https://www.pinterest.com/aesthetics/spring-wallpapers/"
run_vision_cart(test_url)
```

Run it with:

```bash
cd VisionCart
python src/main-LG.py
```

To use a different board, edit `test_url` at line 176 before running. (Argparse is not yet wired up in this variant.)

---

## What gets printed

Each node prints its phase header. On a retry pass you will see procurement run a second (or third) time:

```
[Node] Stylist: Analyzing aesthetic vibes...
--- RAW MODEL OUTPUT ---
{ "style_profile": "...", ... }

[Node] Procurement: Searching products (Attempt 1)...
[procurement] Fetching products for 5 queries...
[procurement] Done. Total candidate pool: 63 products.

[Node] Ranker/Critic: Evaluating product relevance...
[ranker_critic] Results: 2 accepted, 61 rejected
--- Retry: Max score 0.52 <= 0.6. Starting retry 1 ---

[Node] Procurement: Searching products (Attempt 2)...
[procurement] Retry pass — critic feedback: ...
[procurement] Done. Total candidate pool: 71 products.

[Node] Ranker/Critic: Evaluating product relevance...
[ranker_critic] Results: 9 accepted, 62 rejected
--- Success: Found product with score 0.74 > 0.6 ---

[Node] Output: Formatting results...

==================================================
FINAL PIPELINE OUTPUT
==================================================
Your board reads as...
```

If max retries are exhausted, the output begins with `LOW MATCH CONFIDENCE`.

---

## Graph construction

```python
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("stylist", stylist_node)
    workflow.add_node("procurement", procurement_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("stylist")
    workflow.add_edge("stylist", "procurement")
    workflow.add_edge("procurement", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "procurement", "continue": "output"}
    )

    workflow.add_edge("output", END)
    return workflow.compile()
```

The compiled graph is invoked with `app.invoke(initial_state)`, which runs the graph to completion and returns the final state.
