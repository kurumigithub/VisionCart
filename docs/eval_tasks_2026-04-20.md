# VisionCart Evaluation Task List
**Date:** 2026-04-20
**Last verified:** 2026-04-24 — Fix 1 is implemented (`style_profile` ownership moved to stylist-derived state); Fixes 2–3 remain open. `main.py` no longer exists; pipeline entry points are now `main-LC.py` (LangChain) and `main-LG.py` (LangGraph).

---

## Pre-Eval Fixes (Blockers)

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| Fix `style_profile` ownership and handoff in pipeline entry points | `style_profile` should be produced from `stylist_output`, not echoed from procurement. Right now both `main-LC.py` and `main-LG.py` read `style_profile` from `procurement.run()` output, which makes procurement look like the owner of style state and allows schema drift. The ranker expects a structured dict and currently receives an incompatible string path in production. Fix: construct a ranker-compatible `state["style_profile"]` directly after stylist runs, then keep procurement output limited to `candidate_products`, `procurement_queries`, and `iterations`. | ✅ Fixed | [`main-LC.py:60–74`](../src/main-LC.py#L60), [`main-LG.py:14–38`](../src/main-LG.py#L14), [`procurement.py:221–227`](../src/agents/procurement.py#L221), [`ranker_critic.py:395`](../src/agents/ranker_critic.py#L395) |
| Fix ranker rejection threshold for text-only mode | `OVERALL_REJECT_THRESHOLD = 0.35` was designed assuming 50% image similarity weight. In text-only mode the formula reweights to `0.6 × txt_sim + 0.4 × sem_score`. A product with zero keyword overlap scores `0.6×0 + 0.4×0.5 = 0.20`, which is always rejected. 2,784 of 3,534 total rejections hit exactly this floor. Fix: when `has_image_scores` is False, use `effective_threshold = overall_reject_threshold * 0.7` (≈0.245). | ❌ Not fixed | [`ranker_critic.py:24`](../src/agents/ranker_critic.py#L24), [`ranker_critic.py:212–217`](../src/agents/ranker_critic.py#L212), [`ranker_critic.py:183–218`](../src/agents/ranker_critic.py#L183) |
| Expand `STYLE_DEFINITIONS` with product category keywords | Style keywords in `STYLE_DEFINITIONS` are biased toward aesthetic/material terms (`herringbone`, `tweed`, `cashmere`) but missing product category words. Queries like `"leather satchel bag"`, `"platform chunky sneaker"`, `"walnut credenza"`, `"rattan pendant light"`, `"linen curtains"` all score zero text similarity because none of their key nouns appear in style keywords. `dataset.json` already has `product_terms` and `style_terms` per product — these are the missing terms and are not currently used. Fix: pull category terms from `dataset.json` and add per-style to `STYLE_DEFINITIONS`. | ❌ Not fixed | [`test_eval.py:70–312`](../tests/test_eval.py#L70), [`test_eval.py:317–342`](../tests/test_eval.py#L317) |

---

## End-to-End Pipeline Eval

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| Full pipeline runs from Pinterest URL | Run 3–5 end-to-end runs via `main-LC.py` or `main-LG.py` on boards from different aesthetics. Log full state at each agent handoff. Currently hardcoded to a single test URL. Requires API keys + GPU; no longer blocked by fix 1. | ⚠️ Ready after env/API setup | [`main-LC.py:41–84`](../src/main-LC.py#L41), [`main-LG.py:115`](../src/main-LG.py#L115) |
| State continuity validator | Add a validator that asserts required keys are present in state at each agent handoff. `state.py` defines the schema as a TypedDict but nothing enforces it at runtime. Catches regressions like the current style_profile bug silently. | ⚠️ Partial (pre-ranker guard added) | [`state.py:1–56`](../src/graph/state.py#L1), [`main-LC.py:70–74`](../src/main-LC.py#L70), [`main-LG.py:36–38`](../src/main-LG.py#L36) |
| Per-agent latency instrumentation | Wrap each `agent.run()` call in `main-LC.py` with `time.perf_counter()` and log elapsed time. Stylist and output (local models) will dominate; procurement is network-bound. Needed to know where to optimize. ~5 lines to add. | ❌ Not instrumented | [`main-LC.py:59–76`](../src/main-LC.py#L59) |

---

## Cross-Agent Coordination Tasks

Items extracted from agent guides that require team-wide decisions or cross-agent code changes, ordered by dependency.

### P0 — Foundation (no dependencies, unblocks everything else)

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| Canonicalize `style_profile` schema and ownership in `state.py` | The ranker expects a structured dict; `test_eval.py` builds one format; both `main-LC.py` and `main-LG.py` currently route top-level `style_profile` through procurement output. There is no single source of truth. Define a canonical schema (with `style_keywords`, `color_palette`, `materials`, `board_embedding`) in `graph/state.py` or a shared util, and set policy that stylist-derived state is the only writer of `style_profile`. | ✅ Fixed | [`state.py:1–56`](../src/graph/state.py#L1), [`main-LC.py:60–74`](../src/main-LC.py#L60), [`main-LG.py:14–38`](../src/main-LG.py#L14), [`ranker_critic.py:22–46`](../src/agents/ranker_critic.py#L22) |
| Decide whether stylist `products` field influences procurement | The stylist emits a `products` key listing item types it identified in the images (e.g. `"blazer"`, `"planter"`). Procurement does not read it. Decide: add `products` to the procurement prompt to seed category selection, or remove the field from the stylist schema entirely. Either way this determines the final contract between stylist and procurement. | ❌ Decision pending | [`stylist.py:57`](../src/agents/stylist.py#L57), [`procurement.py:33–65`](../src/agents/procurement.py#L33) |
| Decide where `output_text` gets consumed | The output agent writes `output_text` and `output_products`, currently just printed to stdout. Decide the downstream target (UI component, Gradio/Streamlit, notebook, terminal) — this determines how the output agent formats its response and what fields `output_products` needs. | ❌ Decision pending | [`main-LC.py:78–83`](../src/main-LC.py#L78), [`output.py:64–142`](../src/agents/output.py#L64) |

### P1 — Schema consumers (depend on P0 canonical schema)

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| Confirm `ranked_products` key schema across team | The output agent reads `name`, `score`, `tags`, `price`, `url`, `image_url`. The ranker writes exactly these keys. But the output guide's starter code uses a different field mapping than the actual implementation. Confirm alignment before the output agent is finalized. Depends on P0 canonical schema being agreed on. | ⚠️ Potential mismatch | [`ranker_critic.py:411–422`](../src/agents/ranker_critic.py#L411), [`output.py:33–46`](../src/agents/output.py#L33) |
| Uncomment argparse block in `main-LC.py` | The CLI argument parser is commented out at lines 87–98; the board URL is hardcoded. Uncomment to make the pipeline runnable without editing source. No schema dependency but should be done before any real pipeline runs. (`main-LG.py` has the same issue.) | ❌ Commented out | [`main-LC.py:87–98`](../src/main-LC.py#L87) |
| Wire `num_products` through procurement and output | `num_products` is initialized to `5` in state but not used to cap anything — procurement and output both hardcode their own limits. Wire it into `results_per_query` in procurement and the top-K cap in output so it functions as a real control. Depends on output consumption decision (P0) to know the right default. | ❌ Not wired | [`main-LC.py:51`](../src/main-LC.py#L51), [`procurement.py:191`](../src/agents/procurement.py#L191), [`output.py:36`](../src/agents/output.py#L36) |

### P2 — Pipeline integrity (depend on Pre-Eval Fixes 1–3 being applied)

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| State continuity validator | Add assertions in `main-LC.py` that check required state keys are present after each agent call. A silent stylist failure (empty JSON parse) currently causes procurement to crash with a non-obvious `ValueError`. Depends on the canonical schema (P0) to know exactly what keys to assert. | ⚠️ Partial (style_profile guardrails only) | [`state.py:1–56`](../src/graph/state.py#L1), [`main-LC.py:70–74`](../src/main-LC.py#L70), [`main-LG.py:36–38`](../src/main-LG.py#L36) |
| Increase stylist image cap beyond 5 | The stylist hard-caps at 5 images regardless of board size. Evaluate increasing to 8–10 to capture more variety. Meaningful only after Fix 1 is applied so the ranker actually uses the richer style profile the stylist produces. Requires testing on T4 (16 GB VRAM) for memory impact. | ❌ Decision pending | [`stylist.py:43`](../src/agents/stylist.py#L43) |
| Per-agent latency instrumentation | Wrap each `agent.run()` call in `main-LC.py` with `time.perf_counter()`. Meaningful only after the pipeline is end-to-end working (fixes 1–3 applied) — latency numbers are irrelevant if the ranker is silently broken. | ❌ Not instrumented | [`main-LC.py:59–76`](../src/main-LC.py#L59) |

### P3 — Enhanced features (depend on P2 pipeline working correctly)

| Task | Detail | Status | Code Reference |
|------|--------|--------|----------------|
| Implement `critic_feedback` summarization | The ranker writes rejection reasons to `rejected_products` but nothing converts them into the `critic_feedback` string procurement reads on retry. The retry loop exists in code but can never fire. Only worth implementing once fixes 1–3 are applied and rejections are meaningful (not just the 0.20 floor case). | ❌ Not implemented | [`ranker_critic.py:427–428`](../src/agents/ranker_critic.py#L427), [`procurement.py:44–47`](../src/agents/procurement.py#L44) |
| Implement CLIP/SigLIP embedding pipeline | Activates the ranker's 50% image similarity weight, currently dead. Requires computing board embeddings (stylist/preprocessing) and product embeddings (procurement/post-processing) and writing them as float lists into state. Highest-effort task — only worth starting once the text-only pipeline is verified working end-to-end. | ❌ Not implemented | [`ranker_critic.py:237–244`](../src/agents/ranker_critic.py#L237), [`procurement.py:122–133`](../src/agents/procurement.py#L122) |

---

## Quick Reference: What Can Be Done Right Now

| Task | Prerequisite |
|------|-------------|
| Fix rejection threshold (fix 2) | None |
| Fix `style_profile` bug (fix 1) | ✅ Done |
| Fix `STYLE_DEFINITIONS` keywords (fix 3) | None |
| Uncomment argparse block in `main-LC.py` / `main-LG.py` | None |
| Wire `num_products` through procurement and output | None |
| Add latency instrumentation to `main-LC.py` | None |
| Add state continuity validator | None |
| Canonicalize `style_profile` schema in `state.py` | ✅ Done |
| Full pipeline run | Fixes 1–3 + GPU + HF_TOKEN + SERPAPI_API_KEY |
---
