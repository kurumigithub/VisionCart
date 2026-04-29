"""
VisionCart Multi-Variant Evaluation Framework

Compares three system configurations on a fixed evaluation corpus of vision
boards paired with a product pool.  Runs against a frozen validation or test
split and reports standard retrieval metrics plus optional LLM-as-judge scores.

Variants
--------
bm25  — BM25 keyword-only retrieval (no style profile, pure term matching)
lc    — Single-pass ranker (style-aware, no retry loop; mirrors main-LC.py)
lg    — Agentic ranker with critic-guided retry (up to 3 passes; mirrors main-LG.py)

Datasets
--------
validation  — dataset/eval.json  (24 entries: 8 styles × 3 queries)
test        — dataset/test.json  ( 6 entries: 2 styles × 3 queries; separate branch)

Metrics
-------
Precision@K   fraction of top-K results that are ground-truth products
Recall@K      fraction of ground-truth products found in top-K results
MRR           mean reciprocal rank of first ground-truth product in the ranking
CLIP sim      visual cosine similarity between board and product embeddings
              (returns NaN if CLIP embeddings are unavailable — see P3 in eval_tasks)
LLM judge     Claude-as-judge: aesthetic match + explanation quality (1-5 each)
              requires ANTHROPIC_API_KEY; enabled with --llm-judge

Usage
-----
python eval.py --split validation --variant bm25
python eval.py --split validation --variant lc
python eval.py --split test --variant lg
python eval.py --split validation --variant all
python eval.py --split validation --variant bm25,lc --out results/comparison.json
python eval.py --split validation --variant all --llm-judge --k 5
python eval.py --split validation --variant lc --style dark_academia
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.ranker_critic import rank_and_critique  # noqa: E402

DATASET_PATH    = REPO_ROOT / "dataset" / "dataset.json"
EVAL_JSON_PATH  = REPO_ROOT / "dataset" / "eval.json"
TEST_JSON_PATH  = REPO_ROOT / "dataset" / "test.json"


# ── Style definitions (mirrors tests/test_eval.py, extended by Fix 3) ─────────

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
        "materials": {"preferred": ["wool", "tweed", "leather", "velvet", "corduroy"], "avoid": []},
        "constraints": {"must_avoid": []},
        "aesthetic":      ["dark academia", "scholarly vintage", "oxford prep", "literary"],
        "colors":         ["brown", "navy", "forest green", "burgundy"],
        "proc_materials": ["wool", "tweed", "leather", "velvet"],
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
        "materials": {"preferred": ["cashmere", "silk", "merino", "linen", "satin"], "avoid": []},
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
        "materials": {"preferred": ["vinyl", "patent leather", "mesh", "metallic fabric"], "avoid": []},
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
        "materials": {"preferred": ["linen", "cotton", "rattan", "ceramic", "raffia"], "avoid": []},
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
        "materials": {"preferred": ["velvet", "brocade", "silk", "embroidered fabric"], "avoid": []},
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
        "materials": {"preferred": ["leather", "suede", "stone", "raw linen", "matte ceramic"], "avoid": []},
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
        "materials": {"preferred": ["teak", "walnut", "brass", "wool felt", "leather"], "avoid": []},
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
        "materials": {"preferred": ["suede", "fringe", "crochet", "leather", "macrame"], "avoid": []},
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
        "materials": {"preferred": ["linen", "oak", "bamboo", "matte ceramic", "cotton"], "avoid": []},
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
        "materials": {"preferred": ["ceramic", "rattan", "linen", "wood", "cotton"], "avoid": []},
        "constraints": {"must_avoid": []},
        "aesthetic":      ["cottagecore", "pastoral", "garden whimsy", "rural cozy"],
        "colors":         ["sage green", "cream", "dusty rose", "terracotta"],
        "proc_materials": ["ceramic", "rattan", "linen", "wood"],
    },
}


def _make_style_profile(
    style_label: str,
    board_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Build a ranker-compatible style_profile from STYLE_DEFINITIONS.

    board_embeddings: pre-computed CLIP embeddings keyed by style_label.
    When provided, the board_embedding field is populated so the ranker's
    image_similarity component is active (weight 50%).  Without it the ranker
    falls back to text-only mode (weights renormalised to 0.60/0.40).
    """
    defn = STYLE_DEFINITIONS.get(style_label, {})
    return {
        "board_id":        style_label,
        "style_summary":   defn.get("style_summary", ""),
        "style_keywords":  list(defn.get("style_keywords", [])),
        "style_elements":  defn.get("style_elements", []),
        "color_palette":   defn.get("color_palette", {"dominant": [], "accent": [], "avoid": []}),
        "materials":       defn.get("materials", {"preferred": [], "avoid": []}),
        "constraints":     defn.get("constraints", {"must_avoid": []}),
        "board_embedding": (board_embeddings or {}).get(style_label, []),
    }


def _format_product(
    p: Dict[str, Any],
    pid: str,
    product_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    # Strip style-label phrases from tags so BM25 cannot win by matching
    # "dark academia" directly in the tag list — real SerpAPI products wouldn't
    # have those curator-added style labels.  We block full phrases, not
    # individual tokens, so legitimate tags like "dark brown" are kept.
    style_label = p.get("style_label", "")
    blocked_phrases = {
        term.lower().replace("_", " ")
        for term in p.get("style_terms", []) + [style_label]
        if term
    }

    def _clean(terms):
        return [t for t in terms
                if t.lower().replace("_", " ") not in blocked_phrases]

    merged_tags = list(dict.fromkeys(
        _clean(p.get("tags", []))
        + p.get("query_terms", [])
        + p.get("product_terms", [])
    ))
    name_key = (p.get("product_name") or "").lower().strip()
    return {
        "product_id":      pid,
        "product_name":    p.get("product_name", ""),
        "image_url":       p.get("image_url", ""),
        "tags":            merged_tags,
        "link":            p.get("link", ""),
        "price":           None,
        "image_embedding": (product_embeddings or {}).get(name_key, []),
    }


# ── BM25 implementation ───────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r'\b[a-z][a-z0-9]*\b', text.lower()) if len(t) > 1]


class _BM25:
    """Simple BM25 ranking over a fixed product corpus."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized = [_tokenize(t) for t in corpus_texts]
        N = len(self.tokenized)
        self.avgdl = sum(len(d) for d in self.tokenized) / N if N else 1.0
        df: Counter = Counter()
        for doc in self.tokenized:
            for tok in set(doc):
                df[tok] += 1
        self.idf: Dict[str, float] = {
            tok: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def scores(self, query_tokens: List[str]) -> List[float]:
        results = []
        for doc in self.tokenized:
            tf_map = Counter(doc)
            dl = len(doc)
            s = 0.0
            for tok in query_tokens:
                if tok not in self.idf:
                    continue
                tf = tf_map.get(tok, 0)
                s += self.idf[tok] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            results.append(s)
        return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def _product_text(p: Dict[str, Any]) -> str:
    return " ".join([p.get("product_name", "")] + p.get("tags", []))


def _precision_at_k(ranked_ids: List[str], gt_ids: set, k: int) -> float:
    top = ranked_ids[:k]
    return len(set(top) & gt_ids) / k if k else 0.0


def _recall_at_k(ranked_ids: List[str], gt_ids: set, k: int) -> float:
    top = ranked_ids[:k]
    return len(set(top) & gt_ids) / len(gt_ids) if gt_ids else 0.0


def _mrr(ranked_ids: List[str], gt_ids: set) -> float:
    for i, pid in enumerate(ranked_ids, 1):
        if pid in gt_ids:
            return 1.0 / i
    return 0.0


def _clip_similarity(
    board_image_paths: List[str],
    product_image_urls: List[str],
) -> Optional[float]:
    """CLIP cosine similarity stub.

    Returns None until the CLIP/SigLIP embedding pipeline is implemented
    (see P3 in docs/eval_tasks_2026-04-20.md).  When board_embedding and
    product image_embedding are populated in state, replace this stub with a
    real cosine similarity computation.
    """
    return None


def _llm_judge(
    style_label: str,
    query: str,
    ranked_products: List[Dict[str, Any]],
    max_products: int = 5,
) -> Optional[Dict[str, float]]:
    """LLM-as-judge via Claude API.

    Rates the top-K ranked products on:
      aesthetic_match    (1-5): how well products match the described style
      explanation_quality (1-5): clarity and accuracy of reasoning/tags

    Requires ANTHROPIC_API_KEY in the environment.
    Returns None if the API key is missing or the call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    top = ranked_products[:max_products]
    if not top:
        return None

    product_list = "\n".join(
        f"{i+1}. {p.get('name', p.get('product_name', ''))} "
        f"[tags: {', '.join(str(t) for t in p.get('tags', [])[:6])}]"
        for i, p in enumerate(top)
    )

    prompt = f"""You are evaluating a product retrieval system for fashion and home décor discovery.

Style: {style_label.replace('_', ' ')}
User query: {query}

Top retrieved products:
{product_list}

Rate the retrieval on a scale of 1-5 for each criterion:
1. aesthetic_match: Do these products genuinely fit the {style_label.replace('_', ' ')} aesthetic?
2. explanation_quality: Are the product tags/descriptions accurate and informative?

Respond with only valid JSON in this exact format:
{{"aesthetic_match": <1-5>, "explanation_quality": <1-5>, "reasoning": "<one sentence>"}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "aesthetic_match":     float(parsed.get("aesthetic_match", 0)),
                "explanation_quality": float(parsed.get("explanation_quality", 0)),
                "reasoning":           parsed.get("reasoning", ""),
            }
    except Exception:
        pass
    return None


# ── Variant runners ───────────────────────────────────────────────────────────

def run_bm25(
    entry: Dict[str, Any],
    all_products: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """BM25 baseline: rank by keyword match against the style label alone.

    Simulates a simple e-commerce keyword search with no style persona and no
    image analysis — the user just types the style name (e.g. "dark academia").
    This is the lower bound: it finds products tagged with the style but cannot
    discriminate between product types within that style the way LC/LG can.

    Intentionally excludes query_terms (answer key) and the style persona so
    any improvement by LC/LG is attributable solely to the ML-generated persona.
    """
    style_label  = entry["style_label"]
    query_tokens = _tokenize(style_label.replace("_", " "))
    gt_names     = {p["product_name"].lower() for p in entry["ground_truth"]}

    corpus_texts = [_product_text(p) for p in all_products]
    bm25 = _BM25(corpus_texts)
    raw_scores = bm25.scores(query_tokens)

    ranked = sorted(
        range(len(all_products)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )
    ranked_products = [
        {
            **all_products[i],
            "bm25_score": round(raw_scores[i], 4),
            "rank": rank + 1,
        }
        for rank, i in enumerate(ranked)
    ]

    gt_ids  = {p["product_id"] for p in all_products if p["product_name"].lower() in gt_names}
    ranked_ids = [p["product_id"] for p in ranked_products]

    return {
        "variant":         "bm25",
        "style_label":     entry["style_label"],
        "query":           entry["query"],
        "ranked_products": ranked_products[:k],
        "ranked_ids_full": ranked_ids,
        "gt_ids":          gt_ids,
        "pool_size":       len(all_products),
    }


def run_lc(
    entry: Dict[str, Any],
    all_products: List[Dict[str, Any]],
    k: int = 5,
    style_profile: Optional[Dict[str, Any]] = None,
    board_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Single-pass ranker (LC): one call to rank_and_critique, no retry."""
    if style_profile is None:
        style_profile = _make_style_profile(entry["style_label"], board_embeddings)
    gt_names = {p["product_name"].lower() for p in entry["ground_truth"]}
    gt_ids   = {p["product_id"] for p in all_products if p["product_name"].lower() in gt_names}

    result = rank_and_critique(style_profile, all_products)

    accepted = result.accepted_products
    rejected = result.rejected_products

    by_id = {p["product_id"]: p for p in all_products}
    ranked_products = []
    for ap in accepted:
        orig = by_id.get(ap.product_id, {})
        ranked_products.append({
            "product_id":   ap.product_id,
            "product_name": orig.get("product_name", ""),
            "score":        ap.final_score,
            "rank":         ap.rank,
            "tags":         orig.get("tags", []),
            "reason":       ap.reason,
        })

    ranked_ids  = [p["product_id"] for p in ranked_products]
    rejected_ids = [p.product_id for p in rejected]
    # Append rejected products after accepted for complete metric computation
    ranked_ids_full = ranked_ids + rejected_ids

    return {
        "variant":         "lc",
        "style_label":     entry["style_label"],
        "query":           entry["query"],
        "ranked_products": ranked_products[:k],
        "ranked_ids_full": ranked_ids_full,
        "gt_ids":          gt_ids,
        "total_accepted":  len(accepted),
        "total_rejected":  len(rejected),
        "pool_size":       len(all_products),
    }


def run_lg(
    entry: Dict[str, Any],
    all_products: List[Dict[str, Any]],
    k: int = 5,
    max_iterations: int = 3,
    retry_threshold: float = 0.6,
    style_profile: Optional[Dict[str, Any]] = None,
    board_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Agentic ranker (LG): rank_and_critique with critic-guided retry loop.

    On each failed pass (max_score ≤ retry_threshold), the critic feedback is
    used to expand the style_profile's style_keywords with terms from rejected
    products that overlap with the broader style vocabulary.  This simulates
    what procurement would do in the full pipeline: generate new, broader queries
    and retrieve different products.  In the offline setting with a fixed pool
    the candidate set is unchanged, but the ranker's scoring profile widens.
    """
    style_label  = entry["style_label"]
    query        = entry["query"]
    gt_names     = {p["product_name"].lower() for p in entry["ground_truth"]}
    gt_ids       = {p["product_id"] for p in all_products if p["product_name"].lower() in gt_names}

    if style_profile is None:
        style_profile = _make_style_profile(style_label, board_embeddings)
    iteration = 0
    last_result = None
    iteration_log: List[Dict[str, Any]] = []

    while iteration < max_iterations:
        iteration += 1
        result = rank_and_critique(style_profile, all_products)
        accepted = result.accepted_products
        max_score = max((p.final_score for p in accepted), default=0.0)

        iteration_log.append({
            "iteration":      iteration,
            "accepted":       len(accepted),
            "rejected":       len(result.rejected_products),
            "max_score":      round(max_score, 4),
            "retry":          max_score <= retry_threshold and iteration < max_iterations,
        })
        last_result = result

        if max_score > retry_threshold:
            break

        if iteration < max_iterations:
            # Build critic feedback: collect terms from rejected products that
            # belong to the broader style vocabulary and expand style_keywords.
            style_vocab = set(style_profile.get("style_keywords", []))
            extra_terms: set = set()
            for rp in result.rejected_products:
                orig = next((p for p in all_products if p["product_id"] == rp.product_id), {})
                for tag in orig.get("tags", []):
                    tok = tag.lower()
                    if tok not in style_vocab and len(tok) > 2:
                        extra_terms.add(tok)
            if extra_terms:
                expanded = list(style_profile["style_keywords"]) + sorted(extra_terms)
                style_profile = {**style_profile, "style_keywords": expanded}

    by_id = {p["product_id"]: p for p in all_products}
    ranked_products = []
    for ap in (last_result.accepted_products if last_result else []):
        orig = by_id.get(ap.product_id, {})
        ranked_products.append({
            "product_id":   ap.product_id,
            "product_name": orig.get("product_name", ""),
            "score":        ap.final_score,
            "rank":         ap.rank,
            "tags":         orig.get("tags", []),
            "reason":       ap.reason,
        })

    rejected_ids   = [p.product_id for p in (last_result.rejected_products if last_result else [])]
    ranked_ids     = [p["product_id"] for p in ranked_products]
    ranked_ids_full = ranked_ids + rejected_ids

    return {
        "variant":         "lg",
        "style_label":     entry["style_label"],
        "query":           query,
        "ranked_products": ranked_products[:k],
        "ranked_ids_full": ranked_ids_full,
        "gt_ids":          gt_ids,
        "total_accepted":  len(ranked_products),
        "total_rejected":  len(rejected_ids),
        "pool_size":       len(all_products),
        "iterations":      iteration,
        "iteration_log":   iteration_log,
    }


# ── Per-entry metric computation ──────────────────────────────────────────────

def _compute_entry_metrics(
    run_result: Dict[str, Any],
    k: int,
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    gt_ids      = run_result["gt_ids"]
    ranked_full = run_result["ranked_ids_full"]
    ranked_k    = ranked_full[:k]

    return {
        "variant":         run_result["variant"],
        "style_label":     run_result["style_label"],
        "query":           run_result["query"],
        "precision_at_1":  round(_precision_at_k(ranked_full, gt_ids, 1), 4),
        "precision_at_3":  round(_precision_at_k(ranked_full, gt_ids, 3), 4),
        f"precision_at_{k}": round(_precision_at_k(ranked_full, gt_ids, k), 4),
        "recall_at_1":     round(_recall_at_k(ranked_full, gt_ids, 1), 4),
        "recall_at_3":     round(_recall_at_k(ranked_full, gt_ids, 3), 4),
        f"recall_at_{k}":  round(_recall_at_k(ranked_full, gt_ids, k), 4),
        "recall_full":     round(_recall_at_k(ranked_full, gt_ids, len(ranked_full)), 4),
        "mrr":             round(_mrr(ranked_full, gt_ids), 4),
        "clip_similarity": _clip_similarity([], []),
        "llm_judge": (
            _llm_judge(
                run_result["style_label"],
                run_result["query"],
                run_result["ranked_products"],
            )
            if use_llm_judge else None
        ),
        "pool_size":       run_result.get("pool_size"),
        "total_accepted":  run_result.get("total_accepted"),
        "total_rejected":  run_result.get("total_rejected"),
        "iterations":      run_result.get("iterations"),
    }


# ── Aggregation and reporting ─────────────────────────────────────────────────

_METRIC_KEYS = [
    "precision_at_1", "precision_at_3",
    "recall_at_1", "recall_at_3",
    "mrr",
]


def _aggregate(
    entry_metrics: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    if not entry_metrics:
        return {}

    pk_key  = f"precision_at_{k}"
    rk_key  = f"recall_at_{k}"
    all_keys = _METRIC_KEYS + [pk_key, rk_key]

    agg: Dict[str, Any] = {}
    n = len(entry_metrics)

    for key in all_keys:
        vals = [m[key] for m in entry_metrics if m.get(key) is not None]
        agg[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else None

    llm_scores = [m["llm_judge"] for m in entry_metrics if m.get("llm_judge")]
    if llm_scores:
        agg["mean_llm_aesthetic_match"] = round(
            sum(s["aesthetic_match"] for s in llm_scores) / len(llm_scores), 4
        )
        agg["mean_llm_explanation_quality"] = round(
            sum(s["explanation_quality"] for s in llm_scores) / len(llm_scores), 4
        )

    by_style: Dict[str, List[float]] = {}
    for m in entry_metrics:
        mrr_val = m.get("mrr")
        if mrr_val is not None:
            by_style.setdefault(m["style_label"], []).append(mrr_val)
    agg["mrr_by_style"] = {s: round(sum(v) / len(v), 4) for s, v in sorted(by_style.items())}
    agg["total_queries"] = n
    return agg


def _print_comparison_table(
    results_by_variant: Dict[str, Dict[str, Any]],
    k: int,
) -> None:
    pk = f"precision_at_{k}"
    rk = f"recall_at_{k}"
    metric_labels = [
        ("Precision@1",  "mean_precision_at_1"),
        ("Precision@3",  "mean_precision_at_3"),
        (f"Precision@{k}", f"mean_{pk}"),
        ("Recall@1",     "mean_recall_at_1"),
        ("Recall@3",     "mean_recall_at_3"),
        (f"Recall@{k}",  f"mean_{rk}"),

        ("MRR",          "mean_mrr"),
        ("LLM Aesth.",   "mean_llm_aesthetic_match"),
        ("LLM Expl.",    "mean_llm_explanation_quality"),
    ]

    variants = list(results_by_variant.keys())
    col_w = max(12, max(len(v) for v in variants) + 2)
    row_label_w = 18

    header = f"{'Metric':<{row_label_w}}" + "".join(f"{v:>{col_w}}" for v in variants)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("VARIANT COMPARISON")
    print("=" * len(header))
    print(header)
    print(sep)

    for label, key in metric_labels:
        vals = []
        for v in variants:
            agg = results_by_variant[v].get("aggregate", {})
            val = agg.get(key)
            vals.append(f"{val:.4f}" if val is not None else "  N/A  ")
        print(f"{label:<{row_label_w}}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("=" * len(header))

    # Per-style MRR breakdown
    all_styles: set = set()
    for v_data in results_by_variant.values():
        all_styles |= set(v_data.get("aggregate", {}).get("mrr_by_style", {}).keys())

    if all_styles:
        print("\n  MRR by style:")
        for style in sorted(all_styles):
            row = f"    {style:<30}"
            for v in variants:
                mrr_map = results_by_variant[v].get("aggregate", {}).get("mrr_by_style", {})
                val = mrr_map.get(style)
                row += f"  {val:.4f}" if val is not None else "    N/A"
            print(row)


LOG_PATH = REPO_ROOT / "results" / "eval_log.md"


def _build_log_section(
    results_by_variant: Dict[str, Dict[str, Any]],
    args: Any,
    k: int,
    has_embeddings: bool = False,
) -> str:
    from datetime import datetime
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    variants = list(results_by_variant.keys())
    style_note    = f" | style: {args.style}" if getattr(args, "style", None) else ""
    personas_note = " | personas: real" if getattr(args, "personas", None) else ""
    emb_note      = " | clip: yes" if has_embeddings else " | clip: no"
    header = (
        f"## {ts} — eval.py"
        f" | split: {args.split}"
        f" | variants: {', '.join(variants)}"
        f" | k={k}"
        f"{style_note}"
        f"{personas_note}"
        f"{emb_note}"
        f" | queries: {results_by_variant[variants[0]]['aggregate'].get('total_queries', '?')}"
    )

    pk = f"precision_at_{k}"
    rk = f"recall_at_{k}"
    metric_labels = [
        ("Precision@1",    "mean_precision_at_1"),
        ("Precision@3",    "mean_precision_at_3"),
        (f"Precision@{k}", f"mean_{pk}"),
        ("Recall@1",       "mean_recall_at_1"),
        ("Recall@3",       "mean_recall_at_3"),
        (f"Recall@{k}",   f"mean_{rk}"),

        ("MRR",            "mean_mrr"),
        ("LLM Aesth.",     "mean_llm_aesthetic_match"),
        ("LLM Expl.",      "mean_llm_explanation_quality"),
    ]

    # Metrics table
    col_header = "| Metric | " + " | ".join(v.upper() for v in variants) + " |"
    col_sep    = "|--------|" + "|".join("--------" for _ in variants) + "|"
    rows = [col_header, col_sep]
    for label, key in metric_labels:
        cells = []
        for v in variants:
            val = results_by_variant[v]["aggregate"].get(key)
            cells.append(f"{val:.4f}" if val is not None else "N/A")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    # Per-style MRR table
    all_styles: set = set()
    for v_data in results_by_variant.values():
        all_styles |= set(v_data["aggregate"].get("mrr_by_style", {}).keys())

    style_rows: List[str] = []
    if all_styles:
        style_rows = [
            "",
            "**MRR by style**",
            "",
            "| Style | " + " | ".join(v.upper() for v in variants) + " |",
            "|-------|" + "|".join("--------" for _ in variants) + "|",
        ]
        for style in sorted(all_styles):
            cells = []
            for v in variants:
                val = results_by_variant[v]["aggregate"].get("mrr_by_style", {}).get(style)
                cells.append(f"{val:.4f}" if val is not None else "N/A")
            style_rows.append(f"| {style} | " + " | ".join(cells) + " |")

    lines = [header, ""] + rows + style_rows + ["", "---", ""]
    return "\n".join(lines)


def _append_eval_log(
    results_by_variant: Dict[str, Dict[str, Any]],
    args: Any,
    k: int,
    has_embeddings: bool = False,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    section = _build_log_section(results_by_variant, args, k, has_embeddings)
    with open(LOG_PATH, "a") as f:
        f.write(section)
    print(f"\nResults appended to {LOG_PATH}")


def _print_per_entry(metrics: List[Dict[str, Any]], k: int) -> None:
    pk = f"precision_at_{k}"
    rk = f"recall_at_{k}"
    for m in metrics:
        extra = ""
        if m.get("iterations") is not None:
            extra = f"  iters={m['iterations']}"
        print(
            f"  [{m['variant']:4}] {m['style_label']:<28} | {m['query']:<45}"
            f"  MRR={m['mrr']:.2f}  P@{k}={m.get(pk, 0):.2f}"
            f"{extra}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionCart evaluation framework — compare BM25, LC, LG variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["validation", "test"], default="validation",
        help="Dataset split to evaluate (default: validation)",
    )
    parser.add_argument(
        "--variant", default="all",
        help=(
            "Comma-separated list of variants to run: bm25, lc, lg, or 'all'. "
            "Example: --variant bm25,lc  (default: all)"
        ),
    )
    parser.add_argument(
        "--style", metavar="STYLE_LABEL",
        help="Evaluate only entries with this style_label",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="K for Precision@K and Recall@K metrics (default: 5)",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable LLM-as-judge scoring via Claude API (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Write full results to this JSON file",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-entry results during the run",
    )
    parser.add_argument(
        "--personas", metavar="PATH",
        help=(
            "JSON file of pre-computed style personas keyed by style_label. "
            "When provided, LC/LG use the real VLM-generated profile instead of "
            "STYLE_DEFINITIONS.  BM25 always uses the style label only. "
            "Generate this file with the colab_eval.ipynb notebook."
        ),
    )
    parser.add_argument(
        "--embeddings-dir", metavar="PATH", default="data/embeddings",
        help="Directory containing board_embeddings.json and product_embeddings.json (default: data/embeddings)",
    )
    args = parser.parse_args()

    # ── Resolve dataset path ──────────────────────────────────────────────────
    if args.split == "validation":
        data_path = EVAL_JSON_PATH
    else:
        data_path = TEST_JSON_PATH

    if not data_path.exists():
        print(f"ERROR: dataset file not found: {data_path}")
        if args.split == "test":
            print("  (test.json lives on a separate branch — merge or checkout to access it)")
        sys.exit(1)

    with open(data_path) as f:
        eval_entries: List[Dict[str, Any]] = json.load(f)

    if args.style:
        eval_entries = [e for e in eval_entries if e["style_label"] == args.style]
        if not eval_entries:
            print(f"No entries found for style '{args.style}'.")
            sys.exit(1)

    # ── Load CLIP embeddings (if pre-computed) ────────────────────────────────
    emb_dir = REPO_ROOT / args.embeddings_dir
    board_emb_path   = emb_dir / "board_embeddings.json"
    product_emb_path = emb_dir / "product_embeddings.json"

    board_embeddings: Dict[str, List[float]] = {}
    product_embeddings: Dict[str, List[float]] = {}

    if board_emb_path.exists():
        with open(board_emb_path) as f:
            board_embeddings = json.load(f)
        print(f"[eval] Loaded board embeddings for {len(board_embeddings)} styles.")
    else:
        print(f"[eval] No board embeddings found at {board_emb_path} — image similarity disabled.")

    if product_emb_path.exists():
        with open(product_emb_path) as f:
            product_embeddings = json.load(f)
        print(f"[eval] Loaded product embeddings for {len(product_embeddings)} products.")
    else:
        print(f"[eval] No product embeddings found at {product_emb_path} — image similarity disabled.")

    # ── Load full product pool ────────────────────────────────────────────────
    with open(DATASET_PATH) as f:
        raw_products = json.load(f)
    all_products = [_format_product(p, f"ds_{i}", product_embeddings) for i, p in enumerate(raw_products)]

    # ── Load pre-computed personas (--personas flag) ──────────────────────────
    loaded_personas: Optional[Dict[str, Dict[str, Any]]] = None
    if args.personas:
        personas_path = Path(args.personas)
        if not personas_path.exists():
            print(f"ERROR: --personas file not found: {personas_path}")
            sys.exit(1)
        with open(personas_path) as f:
            loaded_personas = json.load(f)
        # Inject board embeddings into each persona profile
        for style, profile in loaded_personas.items():
            if not profile.get("board_embedding"):
                profile["board_embedding"] = board_embeddings.get(style, [])
        print(f"[eval] Loaded {len(loaded_personas)} real style personas from {personas_path}.")

    # ── Resolve variants ──────────────────────────────────────────────────────
    if args.variant.lower() == "all":
        variants = ["bm25", "lc", "lg"]
    else:
        variants = [v.strip().lower() for v in args.variant.split(",")]

    RUNNERS = {"bm25": run_bm25, "lc": run_lc, "lg": run_lg}
    for v in variants:
        if v not in RUNNERS:
            print(f"Unknown variant '{v}'. Choose from: bm25, lc, lg, all")
            sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────────────
    results_by_variant: Dict[str, Any] = {}

    for variant in variants:
        runner = RUNNERS[variant]
        print(f"\n{'='*60}")
        print(f"Running variant: {variant.upper()}  |  split: {args.split}  |  entries: {len(eval_entries)}")
        print("=" * 60)

        entry_metrics: List[Dict[str, Any]] = []
        entry_raw:     List[Dict[str, Any]] = []

        for entry in eval_entries:
            if loaded_personas is not None and variant in ("lc", "lg"):
                # Use pre-computed VLM persona (board_embedding already injected above)
                sp = loaded_personas.get(entry["style_label"])
                run_result = runner(entry, all_products, k=args.k, style_profile=sp)
            elif variant in ("lc", "lg"):
                # Offline: STYLE_DEFINITIONS + pre-computed board embeddings (if any)
                run_result = runner(entry, all_products, k=args.k, board_embeddings=board_embeddings)
            else:
                run_result = runner(entry, all_products, k=args.k)
            metrics    = _compute_entry_metrics(run_result, args.k, args.llm_judge)
            entry_metrics.append(metrics)
            entry_raw.append(run_result)
            if args.verbose:
                _print_per_entry([metrics], args.k)

        agg = _aggregate(entry_metrics, args.k)
        results_by_variant[variant] = {
            "aggregate":     agg,
            "entry_metrics": entry_metrics,
            "entry_raw":     entry_raw,
        }
        print(f"\n[{variant.upper()}] {agg.get('total_queries')} queries | "
              f"MRR={agg.get('mean_mrr', 'N/A')}  "
              f"Recall@{args.k}={agg.get(f'mean_recall_at_{args.k}', 'N/A')}  "
              )

    # ── Print comparison table ────────────────────────────────────────────────
    if len(variants) > 1:
        _print_comparison_table(results_by_variant, args.k)
    else:
        v = variants[0]
        agg = results_by_variant[v]["aggregate"]
        print(f"\n{'='*60}")
        print(f"RESULTS — {v.upper()} ({args.split})")
        print("=" * 60)
        for key, val in agg.items():
            if key == "mrr_by_style":
                continue
            if val is not None:
                print(f"  {key:<35} {val}")
        if agg.get("mrr_by_style"):
            print("\n  MRR by style:")
            for style, mrr in sorted(agg["mrr_by_style"].items()):
                bar = "█" * int(mrr * 20)
                print(f"    {style:<30} {mrr:.4f}  {bar}")
        print("=" * 60)

    # ── Append to eval log ───────────────────────────────────────────────────
    _append_eval_log(results_by_variant, args, args.k,
                     has_embeddings=bool(board_embeddings and product_embeddings))

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # entry_raw contains non-serialisable sets; convert to lists
        serialisable = {}
        for v, data in results_by_variant.items():
            s_entries = []
            for er in data["entry_raw"]:
                s_entries.append({
                    **er,
                    "gt_ids": sorted(er["gt_ids"]),
                    "ranked_ids_full": er["ranked_ids_full"],
                })
            serialisable[v] = {
                "aggregate":     data["aggregate"],
                "entry_metrics": data["entry_metrics"],
                "entry_raw":     s_entries,
            }
        with open(out_path, "w") as f:
            json.dump(
                {
                    "split":    args.split,
                    "variants": variants,
                    "k":        args.k,
                    "results":  serialisable,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
