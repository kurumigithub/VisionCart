"""
VisionCart evaluation harness — procurement + ranker/critic, no stylist needed.

Loads dataset/eval_queries.json (30 entries: one per unique style_label+query pair,
each with 4 ground-truth products).  Builds mock stylist outputs directly from the
style_label and query_terms so the stylist agent is bypassed entirely.

Evaluation modes
----------------
default (ranker, full dataset pool)
    Uses all 120 dataset products as the candidate pool.  Tests whether the
    ranker can correctly identify the 4 ground-truth products out of all 120.
    This is the closest simulation of "does the model grab the right items from
    the database".  No API keys required.

--gt-only
    Uses only the 4 ground-truth products as candidates.  Useful for checking
    whether the ranker accepts known-good items in isolation, independent of
    how well they rank against unrelated products.

--dataset-search  (requires HF_TOKEN only)
    Uses the real procurement LLM (Qwen via HF API) to generate queries from the
    mock stylist output, then searches the 120-product dataset by keyword overlap
    instead of calling SerpAPI.  Tests end-to-end query generation + ranking with
    no external shopping API needed.

--full  (requires SERPAPI_API_KEY + HF_TOKEN)
    Patches procurement.build_queries to return the known query directly (bypassing
    LLM variance), fires real SerpAPI searches, then runs the ranker.  Measures
    whether the ground-truth product names reappear in the live results.

Usage
-----
python tests/test_eval.py                            # full-pool ranker eval
python tests/test_eval.py --gt-only                  # GT-only ranker eval
python tests/test_eval.py --dataset-search           # LLM queries + dataset search
python tests/test_eval.py --style dark_academia      # single style
python tests/test_eval.py --full                     # procurement + ranker (API keys)
python tests/test_eval.py --out results/eval.json    # save per-query results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
# ranker_critic uses `from src.utils.helper import ...` so needs the repo root.
# Adding src/ as well lets `from agents.xxx import` work in both ranker and here.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.ranker_critic import run as ranker_run  # noqa: E402

EVAL_QUERIES_PATH  = REPO_ROOT / "dataset" / "eval_queries.json"
DATASET_PATH       = REPO_ROOT / "dataset" / "dataset.json"


# ── Style definitions ─────────────────────────────────────────────────────────
# Maps each style_label to the fields needed by both the ranker (style_profile)
# and the procurement agent (stylist_output).  query_terms are injected on top
# at call time so keyword matching targets the exact original query.

STYLE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "dark_academia": {
        "style_summary": (
            "Scholarly, vintage-inspired aesthetic with rich earth tones, classic tailoring, "
            "and literary references. Think Oxford libraries and autumn walks."
        ),
        "style_keywords": [
            "herringbone", "tweed", "blazer", "oxford", "argyle", "plaid",
            "leather", "vintage", "scholarly", "classic",
        ],
        "style_elements": ["tailored", "structured", "dark", "rich", "layered"],
        "color_palette": {
            "dominant": ["brown", "navy", "forest green", "burgundy", "black"],
            "accent": ["gold", "cream", "camel"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["wool", "tweed", "leather", "velvet", "corduroy"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":       ["dark academia", "scholarly vintage", "oxford prep", "literary"],
        "colors":          ["brown", "navy", "forest green", "burgundy"],
        "proc_materials":  ["wool", "tweed", "leather", "velvet"],
    },
    "quiet_luxury": {
        "style_summary": (
            "Understated elegance with neutral tones and impeccable tailoring. "
            "No logos, no noise — just quality fabrics and timeless silhouettes."
        ),
        "style_keywords": [
            "cashmere", "silk", "tailored", "minimalist", "elevated",
            "classic", "neutral", "refined", "turtleneck", "trousers",
        ],
        "style_elements": ["clean lines", "minimal", "classic", "luxe", "understated"],
        "color_palette": {
            "dominant": ["cream", "camel", "beige", "ivory", "white"],
            "accent": ["navy", "black", "grey"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["cashmere", "silk", "merino", "linen", "satin"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["quiet luxury", "old money", "minimalist chic", "understated elegance"],
        "colors":         ["cream", "camel", "beige", "navy"],
        "proc_materials": ["cashmere", "silk", "merino", "linen"],
    },
    "y2k_streetwear": {
        "style_summary": (
            "Nostalgic Y2K revival with iridescent surfaces, bubblegum pinks, "
            "chrome accents, and playful maximalism with a futuristic edge."
        ),
        "style_keywords": [
            "Y2K", "iridescent", "holographic", "chrome", "crop",
            "streetwear", "nostalgic", "bubblegum", "butterfly", "mini",
        ],
        "style_elements": ["shiny", "metallic", "crop", "low-rise", "playful", "maximalist"],
        "color_palette": {
            "dominant": ["pink", "silver", "white", "baby blue"],
            "accent": ["electric blue", "chrome", "holographic"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["vinyl", "patent leather", "mesh", "metallic fabric"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["Y2K", "cyber Y2K", "2000s nostalgia", "pop maximalist"],
        "colors":         ["pink", "silver", "baby blue", "white"],
        "proc_materials": ["vinyl", "patent leather", "mesh", "metallic"],
    },
    "coastal_mediterranean": {
        "style_summary": (
            "Breezy Mediterranean lifestyle with sun-bleached whites, terracotta, "
            "and cobalt blue. Natural textures and airy silhouettes evoking a coastal escape."
        ),
        "style_keywords": [
            "linen", "white", "terracotta", "coastal", "mediterranean",
            "breezy", "airy", "natural", "blue", "flowy",
        ],
        "style_elements": ["relaxed", "breezy", "natural", "light", "airy", "coastal"],
        "color_palette": {
            "dominant": ["white", "cream", "sand", "terracotta"],
            "accent": ["cobalt blue", "navy", "sky blue"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["linen", "cotton", "rattan", "ceramic", "raffia"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["coastal", "mediterranean", "beach vacation", "greek island"],
        "colors":         ["white", "terracotta", "cobalt blue", "sand"],
        "proc_materials": ["linen", "ceramic", "rattan", "cotton"],
    },
    "maximalist_eclectic": {
        "style_summary": (
            "Bold and curated chaos — jewel tones, mixed patterns, velvet textures, "
            "and statement pieces layered with intention."
        ),
        "style_keywords": [
            "bold", "pattern", "velvet", "jewel", "eclectic",
            "statement", "maximalist", "mixed", "ornate", "embroidered",
        ],
        "style_elements": ["layered", "patterned", "bold", "colorful", "statement", "ornate"],
        "color_palette": {
            "dominant": ["emerald", "burgundy", "mustard", "royal blue"],
            "accent": ["gold", "copper", "hot pink"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["velvet", "brocade", "silk", "embroidered fabric"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["maximalist", "eclectic", "bold glam", "baroque modern"],
        "colors":         ["emerald", "burgundy", "mustard", "gold"],
        "proc_materials": ["velvet", "brocade", "silk", "embroidered"],
    },
    "dark_moody_organic": {
        "style_summary": (
            "Dark, textural, and organic — deep tones paired with raw natural materials. "
            "Gothic undertones softened by earthy warmth."
        ),
        "style_keywords": [
            "dark", "moody", "organic", "black", "charcoal",
            "leather", "stone", "gothic", "earthy", "raw",
        ],
        "style_elements": ["dark", "textural", "raw", "earthy", "moody", "natural"],
        "color_palette": {
            "dominant": ["black", "charcoal", "deep forest green", "dark burgundy"],
            "accent": ["rust", "copper", "warm grey"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["leather", "suede", "stone", "raw linen", "matte ceramic"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["dark moody", "gothic organic", "dark nature", "moody earth"],
        "colors":         ["black", "charcoal", "deep forest green", "dark burgundy"],
        "proc_materials": ["leather", "suede", "stone", "raw linen"],
    },
    "mid_century_modern": {
        "style_summary": (
            "Clean geometric lines and retro warmth — teak wood, brass accents, "
            "mustard yellows, and functional elegance from the 1950s-60s."
        ),
        "style_keywords": [
            "mid century", "modern", "teak", "walnut", "brass",
            "retro", "geometric", "MCM", "vintage", "danish",
        ],
        "style_elements": ["geometric", "clean lines", "retro", "functional", "organic form"],
        "color_palette": {
            "dominant": ["mustard", "olive", "teak", "rust orange"],
            "accent": ["brass", "black", "cream"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["teak", "walnut", "brass", "wool felt", "leather"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["mid century modern", "MCM", "retro 60s", "danish modern"],
        "colors":         ["mustard", "olive", "teak", "rust orange"],
        "proc_materials": ["teak", "walnut", "brass", "wool"],
    },
    "vintage_bohemian": {
        "style_summary": (
            "Free-spirited vintage meets global boho — fringe, crochet, suede, "
            "warm earth tones, and an eclectic mix of handmade and found pieces."
        ),
        "style_keywords": [
            "vintage", "bohemian", "fringe", "crochet", "suede",
            "boho", "folk", "handmade", "earthy", "artisan",
        ],
        "style_elements": ["layered", "handmade", "folk", "artisan", "free-spirited", "earthy"],
        "color_palette": {
            "dominant": ["rust", "burnt orange", "warm beige", "terracotta"],
            "accent": ["mustard", "forest green", "turquoise"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["suede", "fringe", "crochet", "leather", "macrame"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["vintage bohemian", "boho folk", "70s revival", "artisan handmade"],
        "colors":         ["rust", "burnt orange", "warm beige", "terracotta"],
        "proc_materials": ["suede", "fringe", "crochet", "leather"],
    },
    "japandi": {
        "style_summary": (
            "Calm stripped-back Japandi interior: warm neutrals, clean lines, "
            "wabi-sabi imperfection, natural materials like linen and pale oak, "
            "and intentional minimalism."
        ),
        "style_keywords": [
            "japandi", "wabi-sabi", "zen", "minimal", "linen",
            "oak", "bamboo", "natural", "neutral", "calm",
        ],
        "style_elements": ["minimal", "natural", "calm", "intentional", "clean", "serene"],
        "color_palette": {
            "dominant": ["warm white", "pale wood", "stone grey", "charcoal"],
            "accent": ["sage green", "clay"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["linen", "oak", "bamboo", "matte ceramic", "cotton"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["japandi", "wabi-sabi", "zen minimalist", "nordic calm"],
        "colors":         ["warm white", "pale wood", "stone grey", "charcoal"],
        "proc_materials": ["linen", "oak", "bamboo", "matte ceramic"],
    },
    "cottagecore": {
        "style_summary": (
            "Whimsical rural idyll — handmade ceramics, rattan, dried flowers, "
            "soft pastels and earthy naturals. Garden living meets cozy craft."
        ),
        "style_keywords": [
            "cottagecore", "handmade", "ceramic", "rattan", "dried flowers",
            "garden", "whimsical", "pastoral", "linen", "natural",
        ],
        "style_elements": ["handmade", "natural", "whimsical", "cozy", "garden", "floral"],
        "color_palette": {
            "dominant": ["sage green", "cream", "dusty rose", "warm beige"],
            "accent": ["terracotta", "lavender", "butter yellow"],
            "avoid": [],
        },
        "materials": {
            "preferred": ["ceramic", "rattan", "linen", "wood", "cotton"],
            "avoid": [],
        },
        "constraints": {"must_avoid": []},
        "aesthetic":      ["cottagecore", "pastoral", "garden whimsy", "rural cozy"],
        "colors":         ["sage green", "cream", "dusty rose", "terracotta"],
        "proc_materials": ["ceramic", "rattan", "linen", "wood"],
    },
}


# ── Mock builders ─────────────────────────────────────────────────────────────

def make_style_profile(style_label: str, query_terms: List[str], query: str = "") -> Dict[str, Any]:
    """
    Build a ranker-compatible style_profile dict.

    The query_terms are merged into style_keywords so the ranker's
    keyword_overlap_score rewards products that match the original search query.

    style_summary is left empty intentionally: ranker_critic._compute_scores
    extends the keyword pool with every word in the summary, which dilutes
    txt_sim when using a generic mock summary.  In production the stylist
    generates a tight, image-specific summary — the eval uses style_keywords
    + injected query_terms instead.
    """
    defn = STYLE_DEFINITIONS.get(style_label, {})
    base_kw = list(defn.get("style_keywords", []))
    merged_kw = list(dict.fromkeys(base_kw + [t.lower() for t in query_terms]))
    return {
        "board_id":      f"{style_label}__{query.replace(' ', '_')}",
        "style_summary": "",
        "style_keywords": merged_kw,
        "style_elements": defn.get("style_elements", []),
        "color_palette": defn.get("color_palette", {"dominant": [], "accent": [], "avoid": []}),
        "materials":     defn.get("materials", {"preferred": [], "avoid": []}),
        "constraints":   defn.get("constraints", {"must_avoid": []}),
        "board_embedding": [],  # no image embeddings → text-only scoring path
    }


def make_stylist_output(style_label: str, query_terms: List[str]) -> Dict[str, Any]:
    """
    Build a procurement-compatible stylist_output dict.

    The query terms are embedded in style_profile so the LLM is nudged toward
    generating queries that match the original search terms.
    """
    defn = STYLE_DEFINITIONS.get(style_label, {})
    query_hint = " ".join(query_terms)
    return {
        "style_profile": f"{defn.get('style_summary', style_label)}  Key search terms: {query_hint}.",
        "aesthetic":     defn.get("aesthetic", [style_label]),
        "colors":        defn.get("colors", []),
        "materials":     defn.get("proc_materials", []),
        "budget_max":    None,
        "budget_currency": "USD",
    }


# ── Candidate formatting ──────────────────────────────────────────────────────

def _format_product(p: Dict[str, Any], pid: str) -> Dict[str, Any]:
    return {
        "product_id":   pid,
        "product_name": p["product_name"],
        "image_url":    p.get("image_url", ""),
        "tags":         p.get("tags", []),
        "link":         p.get("link", ""),
        "price":        None,
    }


def _search_dataset(
    query: str,
    all_candidates: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search the dataset by keyword overlap between the query and product text.

    Splits the query into tokens (dropping stop-words under 3 chars), scores
    every product in all_candidates, and returns the top_k with score > 0
    sorted by descending overlap.
    """
    from src.utils.helper import keyword_overlap_score, normalize_text

    query_words = [w for w in normalize_text(query).split() if len(w) > 2]
    if not query_words:
        return []

    scored: List[tuple] = []
    for product in all_candidates:
        product_text = f"{product['product_name']} {' '.join(product.get('tags', []))}"
        score = keyword_overlap_score(query_words, product_text)
        if score > 0:
            scored.append((score, product))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


# ── Ranker evaluation — full dataset pool ─────────────────────────────────────

def eval_ranker_full_pool(
    entry: Dict[str, Any],
    all_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run ranker_critic with ALL 120 dataset products as the candidate pool.

    This is the most realistic offline eval: the ranker must discriminate the
    4 ground-truth products out of the entire product database.  Maps GT products
    back to their dataset index via product_name to determine their pool IDs.
    """
    style_label = entry["style_label"]
    query       = entry["query"]
    query_terms = entry["query_terms"]
    gt_products = entry["ground_truth"]

    gt_names = {p["product_name"].lower() for p in gt_products}
    gt_ids   = {
        c["product_id"]
        for c in all_candidates
        if c["product_name"].lower() in gt_names
    }

    style_profile = make_style_profile(style_label, query_terms, query)
    state = {
        "style_profile":      style_profile,
        "candidate_products": all_candidates,
    }

    result    = ranker_run(state)
    rc_out    = result.get("ranker_critic_output", {})
    accepted  = rc_out.get("accepted_products", [])
    acc_ids   = {p["product_id"] for p in accepted}

    accepted_gt   = acc_ids & gt_ids
    total_accepted = len(accepted)
    recall         = len(accepted_gt) / len(gt_ids) if gt_ids else 0.0

    rank_map  = {p["product_id"]: p["rank"] for p in accepted}
    gt_ranks  = sorted([rank_map[pid] for pid in accepted_gt if pid in rank_map])

    return {
        "style_label":     style_label,
        "query":           query,
        "total_gt":        len(gt_ids),
        "accepted_gt":     len(accepted_gt),
        "total_accepted":  total_accepted,
        "pool_size":       len(all_candidates),
        "recall":          round(recall, 4),
        "gt_ranks":        gt_ranks,
        "ranked_products": result.get("ranked_products", []),
        "rejected":        result.get("rejected_products", []),
    }


# ── Ranker evaluation — GT-only pool ─────────────────────────────────────────

def eval_ranker_gt_only(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run ranker_critic using only the 4 ground-truth products as candidates.

    Useful for checking whether the ranker accepts known-good items in isolation,
    independent of how they compete with unrelated products.
    """
    style_label = entry["style_label"]
    query       = entry["query"]
    query_terms = entry["query_terms"]
    gt_products = entry["ground_truth"]

    style_profile = make_style_profile(style_label, query_terms, query)
    candidates    = [_format_product(p, f"gt_{i}") for i, p in enumerate(gt_products)]

    state = {
        "style_profile":      style_profile,
        "candidate_products": candidates,
    }

    result   = ranker_run(state)
    rc_out   = result.get("ranker_critic_output", {})
    accepted = rc_out.get("accepted_products", [])
    acc_ids  = {p["product_id"] for p in accepted}
    gt_ids   = {f"gt_{i}" for i in range(len(gt_products))}

    accepted_gt = acc_ids & gt_ids
    recall      = len(accepted_gt) / len(gt_ids) if gt_ids else 0.0
    rank_map    = {p["product_id"]: p["rank"] for p in accepted}
    gt_ranks    = [rank_map.get(gid) for gid in sorted(gt_ids)]

    return {
        "style_label":     style_label,
        "query":           query,
        "total_gt":        len(gt_ids),
        "accepted_gt":     len(accepted_gt),
        "recall":          round(recall, 4),
        "gt_ranks":        gt_ranks,
        "ranked_products": result.get("ranked_products", []),
        "rejected":        result.get("rejected_products", []),
    }


# ── Full-pipeline evaluation (procurement + ranker) ───────────────────────────

def eval_full_pipeline(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run procurement (bypassing LLM query gen) + ranker on one eval entry.

    Patches procurement.build_queries to return the known query directly so we
    test SerpAPI retrieval quality without LLM variance.  Requires SERPAPI_API_KEY
    and HF_TOKEN in the environment.

    Recall is measured by product_name match (no stable IDs across API runs).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass

    from agents import procurement

    style_label = entry["style_label"]
    query       = entry["query"]
    query_terms = entry["query_terms"]
    gt_products = entry["ground_truth"]

    _original_build = procurement.build_queries

    def _fixed_build(style, num_queries, critic_feedback=None):  # noqa: ANN001
        return [query]

    procurement.build_queries = _fixed_build
    try:
        proc_raw  = procurement.run({"stylist_output": make_stylist_output(style_label, query_terms)})
        proc_data = json.loads(proc_raw)
    finally:
        procurement.build_queries = _original_build

    style_profile = make_style_profile(style_label, query_terms, query)
    state = {
        "style_profile":        style_profile,
        "procurement_products": proc_data["procurement_products"],
    }
    result = ranker_run(state)

    gt_names  = {p["product_name"].lower() for p in gt_products}
    found_gt  = [p for p in result.get("ranked_products", []) if p["name"].lower() in gt_names]
    retrieved = len(proc_data["procurement_products"])
    accepted  = len(result.get("ranked_products", []))
    recall    = len(found_gt) / len(gt_products) if gt_products else 0.0

    return {
        "style_label":      style_label,
        "query":            query,
        "total_gt":         len(gt_products),
        "found_in_results": len(found_gt),
        "total_retrieved":  retrieved,
        "total_accepted":   accepted,
        "recall":           round(recall, 4),
        "ranked_products":  result.get("ranked_products", []),
    }


# ── Dataset-search evaluation (LLM queries + local dataset search) ───────────

def eval_dataset_search(
    entry: Dict[str, Any],
    all_candidates: List[Dict[str, Any]],
    num_queries: int = 3,
    top_k_per_query: int = 10,
) -> Dict[str, Any]:
    """
    Full procurement + ranker eval using the dataset as the product database.

    1. Calls procurement.build_queries (real Qwen LLM via HF API) to generate
       search queries from the mock stylist output.
    2. For each generated query, searches all 120 dataset products by keyword
       overlap (no SerpAPI call needed).
    3. Deduplicates and runs ranker_critic on the retrieved candidates.
    4. Reports which GT products were found and how they ranked.

    Requires HF_TOKEN in the environment (or .env at repo root).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass

    from agents.procurement import StylistOutput, build_queries

    style_label = entry["style_label"]
    query       = entry["query"]
    query_terms = entry["query_terms"]
    gt_products = entry["ground_truth"]

    stylist_dict = make_stylist_output(style_label, query_terms)
    style = StylistOutput(
        style_profile=stylist_dict["style_profile"],
        aesthetic=stylist_dict["aesthetic"],
        colors=stylist_dict["colors"],
        materials=stylist_dict["materials"],
    )

    print(f"\n[dataset-search] Generating queries for '{style_label}' / '{query}'...")
    generated_queries = build_queries(style, num_queries=num_queries)
    print(f"[dataset-search] LLM queries: {generated_queries}")

    seen_ids: set = set()
    candidates: List[Dict[str, Any]] = []
    retrieval_log: List[Dict[str, Any]] = []

    for q in generated_queries:
        hits = _search_dataset(q, all_candidates, top_k=top_k_per_query)
        new_hits = [h for h in hits if h["product_id"] not in seen_ids]
        for h in new_hits:
            seen_ids.add(h["product_id"])
            candidates.append(h)
        retrieval_log.append({"query": q, "hits": len(hits), "new": len(new_hits)})
        print(f"[dataset-search]   '{q}' → {len(hits)} hits, {len(new_hits)} new")

    print(f"[dataset-search] Total candidates for ranker: {len(candidates)}")

    if not candidates:
        return {
            "style_label":       style_label,
            "query":             query,
            "generated_queries": generated_queries,
            "retrieval_log":     retrieval_log,
            "total_gt":          len(gt_products),
            "accepted_gt":       0,
            "total_retrieved":   0,
            "total_accepted":    0,
            "recall":            0.0,
            "gt_ranks":          [],
            "ranked_products":   [],
            "rejected":          [],
        }

    gt_names = {p["product_name"].lower() for p in gt_products}
    gt_ids   = {c["product_id"] for c in candidates if c["product_name"].lower() in gt_names}

    style_profile = make_style_profile(style_label, query_terms, query)
    state = {
        "style_profile":      style_profile,
        "candidate_products": candidates,
    }
    result   = ranker_run(state)
    rc_out   = result.get("ranker_critic_output", {})
    accepted = rc_out.get("accepted_products", [])
    acc_ids  = {p["product_id"] for p in accepted}

    accepted_gt = acc_ids & gt_ids
    recall      = len(accepted_gt) / len(gt_products) if gt_products else 0.0
    rank_map    = {p["product_id"]: p["rank"] for p in accepted}
    gt_ranks    = sorted([rank_map[pid] for pid in accepted_gt if pid in rank_map])

    return {
        "style_label":       style_label,
        "query":             query,
        "generated_queries": generated_queries,
        "retrieval_log":     retrieval_log,
        "total_gt":          len(gt_products),
        "accepted_gt":       len(accepted_gt),
        "total_retrieved":   len(candidates),
        "total_accepted":    len(accepted),
        "recall":            round(recall, 4),
        "gt_ranks":          gt_ranks,
        "ranked_products":   result.get("ranked_products", []),
        "rejected":          result.get("rejected_products", []),
    }


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _print_entry_result(r: Dict[str, Any], mode: str) -> None:
    print(f"\n[{mode}] {r['style_label']}  |  '{r['query']}'")
    if mode == "procurement+ranker":
        print(f"  retrieved={r['total_retrieved']}  accepted={r['total_accepted']}"
              f"  GT found={r['found_in_results']}/{r['total_gt']}  recall={r['recall']:.2f}")
    elif mode == "dataset-search":
        print(f"  llm_queries={r['generated_queries']}")
        for log in r.get("retrieval_log", []):
            print(f"    '{log['query']}' → {log['hits']} hits, {log['new']} new")
        print(f"  retrieved={r['total_retrieved']}  accepted={r['total_accepted']}"
              f"  accepted_gt={r['accepted_gt']}/{r['total_gt']}"
              f"  recall={r['recall']:.2f}  gt_ranks={r['gt_ranks']}")
    elif mode == "ranker (full pool)":
        print(f"  pool={r['pool_size']}  accepted_gt={r['accepted_gt']}/{r['total_gt']}"
              f"  total_accepted={r['total_accepted']}  recall={r['recall']:.2f}"
              f"  gt_ranks={r['gt_ranks']}")
    else:
        print(f"  accepted_gt={r['accepted_gt']}/{r['total_gt']}"
              f"  recall={r['recall']:.2f}  gt_ranks={r['gt_ranks']}")
    for rej in r.get("rejected", []):
        if rej.get("product_id", "").startswith("gt_") or mode == "ranker (GT only)":
            print(f"  REJECTED {rej['product_id']}: {rej['reason']}")


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total    = len(results)
    perfect  = sum(1 for r in results if r["recall"] == 1.0)
    mean_rec = sum(r["recall"] for r in results) / total if total else 0.0

    by_style: Dict[str, List[float]] = {}
    for r in results:
        by_style.setdefault(r["style_label"], []).append(r["recall"])
    style_recalls = {s: round(sum(v) / len(v), 4) for s, v in sorted(by_style.items())}

    return {
        "total_queries":  total,
        "perfect_recall": perfect,
        "mean_recall":    round(mean_rec, 4),
        "style_recalls":  style_recalls,
    }


def _print_summary(agg: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated : {agg['total_queries']}")
    print(f"  Perfect recall    : {agg['perfect_recall']} / {agg['total_queries']}")
    print(f"  Mean recall       : {agg['mean_recall']:.4f}")
    print("\n  Recall by style:")
    for style, rec in agg["style_recalls"].items():
        bar = "█" * int(rec * 20)
        print(f"    {style:<28} {rec:.2f}  {bar}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VisionCart eval harness")
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Use only the 4 GT products as candidates (default: all 120 dataset products)",
    )
    parser.add_argument(
        "--dataset-search", action="store_true",
        help="LLM query generation + local dataset keyword search (requires HF_TOKEN only)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full procurement + ranker pipeline (requires SERPAPI_API_KEY + HF_TOKEN)",
    )
    parser.add_argument(
        "--style", metavar="STYLE_LABEL",
        help="Evaluate only this style (e.g. dark_academia)",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Write per-query results to this JSON file",
    )
    args = parser.parse_args()

    with open(EVAL_QUERIES_PATH) as f:
        eval_queries: List[Dict[str, Any]] = json.load(f)

    if args.style:
        eval_queries = [e for e in eval_queries if e["style_label"] == args.style]
        if not eval_queries:
            print(f"No entries found for style '{args.style}'.")
            sys.exit(1)

    # Pre-load full dataset for modes that need it (full-pool, dataset-search)
    all_candidates: Optional[List[Dict[str, Any]]] = None
    if not args.full and not args.gt_only:
        with open(DATASET_PATH) as f:
            raw = json.load(f)
        all_candidates = [_format_product(p, f"ds_{i}") for i, p in enumerate(raw)]

    results = []
    for entry in eval_queries:
        if args.full:
            mode = "procurement+ranker"
            r = eval_full_pipeline(entry)
        elif args.dataset_search:
            mode = "dataset-search"
            r = eval_dataset_search(entry, all_candidates)
        elif args.gt_only:
            mode = "ranker (GT only)"
            r = eval_ranker_gt_only(entry)
        else:
            mode = "ranker (full pool)"
            r = eval_ranker_full_pool(entry, all_candidates)

        _print_entry_result(r, mode)
        results.append(r)

    agg = _aggregate(results)
    _print_summary(agg)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": agg, "results": results}, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
