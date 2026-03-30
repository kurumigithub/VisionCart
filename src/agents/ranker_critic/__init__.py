"""Ranker/Critic combined agent for filtering and ranking products."""

from .agent import rank_and_critique, run
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

__all__ = [
    "rank_and_critique",
    "run",
    "AcceptedProduct",
    "RejectedProduct",
    "RankerCriticOutput",
    "ScoreBreakdown",
    "WEIGHT_IMAGE_SIMILARITY",
    "WEIGHT_TEXT_SIMILARITY",
    "WEIGHT_SEMANTIC_MATCH",
    "IMAGE_SIMILARITY_REJECT_FLOOR",
    "OVERALL_REJECT_THRESHOLD",
    "STRONG_CANDIDATE_THRESHOLD",
    "LIKELY_CANDIDATE_THRESHOLD",
    "AVOID_PENALTY",
    "PREFERRED_BONUS",
    "MUST_AVOID_AUTO_REJECT",
]
