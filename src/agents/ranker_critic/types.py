"""Data types for Ranker/Critic agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ScoreBreakdown:
    image_similarity: float = 0.0
    text_similarity: float = 0.0
    semantic_match_score: float = 0.0


@dataclass
class AcceptedProduct:
    product_id: str = ""
    rank: int = 0
    final_score: float = 0.0
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedProduct:
    product_id: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankerCriticOutput:
    board_id: str = ""
    accepted_products: List[AcceptedProduct] = field(default_factory=list)
    rejected_products: List[RejectedProduct] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board_id": self.board_id,
            "accepted_products": [p.to_dict() for p in self.accepted_products],
            "rejected_products": [p.to_dict() for p in self.rejected_products],
        }
