"""Ranker/Critic configuration — thresholds, weights, and scoring parameters."""

# Scoring weights (must sum to 1.0)
WEIGHT_IMAGE_SIMILARITY: float = 0.50
WEIGHT_TEXT_SIMILARITY: float = 0.30
WEIGHT_SEMANTIC_MATCH: float = 0.20

# Critic rejection thresholds
IMAGE_SIMILARITY_REJECT_FLOOR: float = 0.25
OVERALL_REJECT_THRESHOLD: float = 0.35

# Candidate tiers (informational — used in reason strings)
STRONG_CANDIDATE_THRESHOLD: float = 0.65
LIKELY_CANDIDATE_THRESHOLD: float = 0.45

# Semantic / rule-based penalties & boosts
AVOID_PENALTY: float = 0.25
PREFERRED_BONUS: float = 0.10
MUST_AVOID_AUTO_REJECT: bool = True
