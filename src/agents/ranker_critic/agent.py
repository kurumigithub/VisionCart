"""
Ranker/Critic combined agent
=============================

Accepts a **style profile** (from the Stylist agent) and a list of
**candidate products** (from the Procurement agent), then:

1. **Critic stage** — filters obvious mismatches
2. **Ranker stage** — scores remaining products via image similarity,
   text overlap, and semantic rule-based matching.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from utils.helper import (
    combine_product_text,
    cosine_similarity,
    keyword_overlap_score,
    normalize_text,
)

from .config import (
    AVOID_PENALTY,
    IMAGE_SIMILARITY_REJECT_FLOOR,
    LIKELY_CANDIDATE_THRESHOLD,
    MUST_AVOID_AUTO_REJECT,
    OVERALL_REJECT_THRESHOLD,
    PREFERRED_BONUS,
    STRONG_CANDIDATE_THRESHOLD,
    WEIGHT_IMAGE_SIMILARITY,
    WEIGHT_SEMANTIC_MATCH,
    WEIGHT_TEXT_SIMILARITY,
)
from .types import (
    AcceptedProduct,
    RankerCriticOutput,
    RejectedProduct,
    ScoreBreakdown,
)


# ── helpers ──────────────────────────────────────────────────────────

def _cfg(state: Dict[str, Any], key: str, default: float) -> float:
    """Resolve a config value: state['config'][key] > module-level default."""
    overrides = state.get("config") or {}
    val = overrides.get(key)
    return float(val) if val is not None else default


def _list_lower(seq: Optional[Sequence[str]]) -> List[str]:
    if not seq:
        return []
    return [s.lower() for s in seq if s]


def _ensure_dict(value: Any) -> Dict[str, Any]:
    """Accept a dict or a JSON string; always return a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _ensure_list(value: Any) -> List[Dict[str, Any]]:
    """Accept a list or a JSON string; always return a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


# ── Critic ───────────────────────────────────────────────────────────

def _check_must_avoid(product: dict, constraints: dict) -> Optional[str]:
    """Return a rejection reason if a must_avoid constraint is violated."""
    must_avoid = _list_lower(constraints.get("must_avoid", []))
    if not must_avoid:
        return None
    product_text = combine_product_text(product).lower()
    for term in must_avoid:
        if term in product_text:
            return f"Rejected: must-avoid constraint violated ('{term}' found in product)."
    return None


def _check_avoid_lists(product: dict, color_palette: dict, materials: dict) -> List[str]:
    """Return list of avoid-list violations found in the product."""
    violations: List[str] = []
    product_text = combine_product_text(product).lower()
    for term in _list_lower(color_palette.get("avoid", [])):
        if term in product_text:
            violations.append(f"avoid-color '{term}'")
    for term in _list_lower(materials.get("avoid", [])):
        if term in product_text:
            violations.append(f"avoid-material '{term}'")
    return violations


def _semantic_match_score(
    product: dict,
    style_profile: dict,
    *,
    avoid_penalty: float = AVOID_PENALTY,
    preferred_bonus: float = PREFERRED_BONUS,
) -> float:
    """Rule-based semantic score in [0, 1]."""
    score = 0.5
    color_palette = style_profile.get("color_palette") or {}
    materials_info = style_profile.get("materials") or {}
    style_keywords = _list_lower(style_profile.get("style_keywords", []))
    style_elements = _list_lower(style_profile.get("style_elements", []))
    product_text = combine_product_text(product).lower()

    for term in _list_lower(materials_info.get("preferred", [])):
        if term in product_text:
            score += preferred_bonus
    for term in _list_lower(color_palette.get("dominant", [])):
        if term in product_text:
            score += preferred_bonus
    for term in _list_lower(color_palette.get("accent", [])):
        if term in product_text:
            score += preferred_bonus * 0.5

    all_keywords = style_keywords + style_elements
    if all_keywords:
        kw_hit_rate = keyword_overlap_score(all_keywords, product_text)
        score += kw_hit_rate * 0.2

    violations = _check_avoid_lists(product, color_palette, materials_info)
    score -= len(violations) * avoid_penalty
    return max(0.0, min(1.0, score))


def _critic_evaluate(
    product: dict,
    style_profile: dict,
    image_sim: float,
    semantic_score: float,
    *,
    img_reject_floor: float,
    overall_reject_threshold: float,
    must_avoid_auto_reject: bool,
    w_img: float,
    w_txt: float,
    w_sem: float,
    text_sim: float,
    has_image_scores: bool,
) -> Optional[str]:
    """Return rejection reason or None if product passes."""
    constraints = style_profile.get("constraints") or {}
    if must_avoid_auto_reject:
        reason = _check_must_avoid(product, constraints)
        if reason:
            return reason
    if has_image_scores and image_sim < img_reject_floor:
        return f"Rejected: image similarity too low ({image_sim:.2f} < {img_reject_floor:.2f})."
    color_palette = style_profile.get("color_palette") or {}
    materials_info = style_profile.get("materials") or {}
    violations = _check_avoid_lists(product, color_palette, materials_info)
    if len(violations) >= 2:
        return f"Rejected: multiple avoid-list violations ({', '.join(violations)})."
    if has_image_scores:
        final = w_img * image_sim + w_txt * text_sim + w_sem * semantic_score
    else:
        non_img = w_txt + w_sem
        if non_img > 0:
            final = (w_txt / non_img) * text_sim + (w_sem / non_img) * semantic_score
        else:
            final = 0.0
    if final < overall_reject_threshold:
        return f"Rejected: overall score too low ({final:.2f} < {overall_reject_threshold:.2f})."
    return None


# ── Ranker ───────────────────────────────────────────────────────────

class _ScoringResult:
    """Internal: scores plus metadata about what was computable."""
    def __init__(self, breakdown: ScoreBreakdown, has_image_scores: bool = True):
        self.breakdown = breakdown
        self.has_image_scores = has_image_scores


def _compute_scores(
    product: dict,
    style_profile: dict,
    board_embedding: Sequence[float],
    *,
    avoid_penalty: float = AVOID_PENALTY,
    preferred_bonus: float = PREFERRED_BONUS,
) -> _ScoringResult:
    """Compute the three scoring dimensions for one product."""
    product_embedding = product.get("image_embedding") or []
    has_image = bool(board_embedding and product_embedding)
    if has_image:
        try:
            img_sim = cosine_similarity(board_embedding, product_embedding)
        except (TypeError, ValueError):
            img_sim = 0.0
            has_image = False
    else:
        img_sim = 0.0

    style_keywords = style_profile.get("style_keywords") or []
    style_summary = style_profile.get("style_summary") or ""
    combined_keywords = list(style_keywords)
    if style_summary:
        combined_keywords.extend(
            w for w in normalize_text(style_summary).split() if len(w) > 3
        )
    product_text = combine_product_text(product)
    txt_sim = keyword_overlap_score(combined_keywords, product_text) if combined_keywords else 0.0

    sem_score = _semantic_match_score(
        product, style_profile,
        avoid_penalty=avoid_penalty,
        preferred_bonus=preferred_bonus,
    )

    return _ScoringResult(
        breakdown=ScoreBreakdown(
            image_similarity=round(img_sim, 4),
            text_similarity=round(txt_sim, 4),
            semantic_match_score=round(sem_score, 4),
        ),
        has_image_scores=has_image,
    )


def _build_reason(score: ScoreBreakdown, final: float) -> str:
    """Generate short human-readable reason for accepting a product."""
    parts: List[str] = []
    if final >= STRONG_CANDIDATE_THRESHOLD:
        parts.append("Strong match")
    elif final >= LIKELY_CANDIDATE_THRESHOLD:
        parts.append("Good match")
    else:
        parts.append("Moderate match")
    details: List[str] = []
    if score.image_similarity >= 0.6:
        details.append("high visual similarity")
    if score.text_similarity >= 0.4:
        details.append("good keyword alignment")
    if score.semantic_match_score >= 0.6:
        details.append("strong style/material fit")
    if details:
        parts.append(f"({', '.join(details)})")
    return " ".join(parts) + "."


# ── Public API ───────────────────────────────────────────────────────

def rank_and_critique(
    style_profile: Dict[str, Any],
    candidate_products: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> RankerCriticOutput:
    """Core logic for ranking and filtering products."""
    cfg_state: Dict[str, Any] = {"config": config or {}}
    w_img = _cfg(cfg_state, "weight_image_similarity", WEIGHT_IMAGE_SIMILARITY)
    w_txt = _cfg(cfg_state, "weight_text_similarity", WEIGHT_TEXT_SIMILARITY)
    w_sem = _cfg(cfg_state, "weight_semantic_match", WEIGHT_SEMANTIC_MATCH)
    img_reject_floor = _cfg(cfg_state, "image_similarity_reject_floor", IMAGE_SIMILARITY_REJECT_FLOOR)
    overall_reject = _cfg(cfg_state, "overall_reject_threshold", OVERALL_REJECT_THRESHOLD)
    avoid_penalty = _cfg(cfg_state, "avoid_penalty", AVOID_PENALTY)
    preferred_bonus = _cfg(cfg_state, "preferred_bonus", PREFERRED_BONUS)
    must_avoid_reject = bool((config or {}).get("must_avoid_auto_reject", MUST_AVOID_AUTO_REJECT))

    board_id = style_profile.get("board_id", "")
    board_embedding = style_profile.get("board_embedding") or []

    accepted: List[AcceptedProduct] = []
    rejected: List[RejectedProduct] = []

    for product in candidate_products:
        pid = product.get("product_id", "")
        scoring = _compute_scores(
            product, style_profile, board_embedding,
            avoid_penalty=avoid_penalty,
            preferred_bonus=preferred_bonus,
        )
        scores = scoring.breakdown

        rejection_reason = _critic_evaluate(
            product, style_profile, scores.image_similarity, scores.semantic_match_score,
            img_reject_floor=img_reject_floor, overall_reject_threshold=overall_reject,
            must_avoid_auto_reject=must_avoid_reject,
            w_img=w_img, w_txt=w_txt, w_sem=w_sem,
            text_sim=scores.text_similarity, has_image_scores=scoring.has_image_scores,
        )
        if rejection_reason:
            rejected.append(RejectedProduct(product_id=pid, reason=rejection_reason))
            continue

        if scoring.has_image_scores:
            final = w_img * scores.image_similarity + w_txt * scores.text_similarity + w_sem * scores.semantic_match_score
        else:
            non_img = w_txt + w_sem
            if non_img > 0:
                final = (w_txt / non_img) * scores.text_similarity + (w_sem / non_img) * scores.semantic_match_score
            else:
                final = 0.0
        final = round(final, 4)

        accepted.append(AcceptedProduct(
            product_id=pid,
            final_score=final,
            score_breakdown=scores,
            reason=_build_reason(scores, final),
        ))

    accepted.sort(key=lambda p: p.final_score, reverse=True)
    for i, p in enumerate(accepted, 1):
        p.rank = i

    return RankerCriticOutput(
        board_id=board_id,
        accepted_products=accepted,
        rejected_products=rejected,
    )


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node entry-point."""
    raw_profile = state.get("style_profile") or {}
    style_profile = _ensure_dict(raw_profile)
    raw_products = state.get("candidate_products") or []
    candidate_products = _ensure_list(raw_products)
    config = state.get("config")

    result = rank_and_critique(style_profile, candidate_products, config)
    output = result.to_dict()

    ranked_for_output: List[Dict[str, Any]] = []
    for ap, orig in zip(result.accepted_products, _match_originals(result.accepted_products, candidate_products)):
        ranked_for_output.append({
            "name": orig.get("title") or orig.get("product_name", ""),
            "score": ap.final_score,
            "tags": (orig.get("attributes") or {}).get("style_tags", []),
            "price": orig.get("price"),
            "url": orig.get("link") or orig.get("product_url", ""),
            "image_url": orig.get("image_url", ""),
        })

    return {
        "ranked_products": ranked_for_output,
        "rejected_products": output["rejected_products"],
        "ranker_critic_output": output,
    }


def _match_originals(
    accepted: List[AcceptedProduct],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return original product dicts in the same order as accepted."""
    by_id = {p.get("product_id", ""): p for p in candidates}
    return [by_id.get(a.product_id, {}) for a in accepted]
