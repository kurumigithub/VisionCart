from typing import TypedDict, List, Dict, Any


class StyleProfile(TypedDict, total=False):
    style_keywords: List[str]
    color_palette: List[str]
    materials: List[str]
    board_embedding: List[float]


def _normalize_terms(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    seen = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        clean = raw.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def style_profile_from_stylist_output(stylist_output: Dict[str, Any]) -> StyleProfile:
    return {
        "style_keywords": _normalize_terms(stylist_output.get("aesthetic")),
        "color_palette": _normalize_terms(stylist_output.get("colors")),
        "materials": _normalize_terms(stylist_output.get("materials")),
    }


class AgentState(TypedDict, total=False):
    iterations: int

    # Inputs
    vision_board_paths: List[str]
    num_products: int

    # stylist
    stylist_output: Dict[str, Any]
    style_profile: StyleProfile

    # procurement
    procurement_queries: List[str]
    candidate_products: List[Dict[str, Any]]

    # ranker/critic
    ranked_products: List[Dict[str, Any]]
    rejected_products: List[Dict[str, Any]]
    ranker_critic_output: Dict[str, Any]

    # final output
    output_text: str
    output_products: List[Dict[str, Any]]