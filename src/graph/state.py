from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    interations: int
    
    # Inputs
    vision_board_paths: List[str]
    num_products: int
    
    # stylist
    stylist_output: Dict[str, Any]
    
    # procurement
    procurement_queries: List[str]
    candidate_products: List[Dict[str, Any]]
    style_profile: str
    
    # ranker/critic
    ranked_products: List[Dict[str, Any]]
    rejected_products: List[Dict[str, Any]]
    
    # final output
    output_text: str
    output_products: List[Dict[str, Any]]