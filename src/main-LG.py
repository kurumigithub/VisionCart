import os
import json
import sys
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END

# Import your existing components
from src.graph.state import AgentState
from src.agents import stylist, procurement, ranker_critic, output
from src.utils import pinterest_crawler


def _style_profile_from_stylist(stylist_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build a ranker-compatible style_profile dict from stylist_output.

    procurement.run() returns style_profile as a plain narrative string, which
    ranker_critic._ensure_dict() cannot parse and silently returns {}.  This
    function builds the structured dict the ranker actually needs from the
    richer stylist_output that is already in state.
    """
    aesthetic = stylist_output.get("aesthetic") or []
    colors    = stylist_output.get("colors")    or []
    materials = stylist_output.get("materials") or []
    narrative = stylist_output.get("style_profile", "")
    return {
        "board_id":       "",
        "style_summary":  narrative,
        "style_keywords": list(dict.fromkeys(aesthetic + colors)),
        "style_elements": aesthetic,
        "color_palette": {
            "dominant": colors,
            "accent":   [],
            "avoid":    [],
        },
        "materials": {
            "preferred": materials,
            "avoid":     [],
        },
        "constraints":    {"must_avoid": []},
        "board_embedding": [],
    }

# ─── Node Functions ──────────────────────────────────────────────────

def stylist_node(state: AgentState):
    print("\n[Node] Stylist: Analyzing aesthetic vibes...")
    return stylist.run(state)

def procurement_node(state: AgentState):
    current_iter = state.get("iterations", 0)
    print(f"\n[Node] Procurement: Searching products (Attempt {current_iter + 1})...")
    
    # Run procurement
    procurement_raw = procurement.run(state)
    procurement_data = json.loads(procurement_raw)
    
    return {
        "candidate_products": procurement_data["procurement_products"],
        "procurement_queries": procurement_data["procurement_queries"],
        "style_profile": _style_profile_from_stylist(state["stylist_output"]),
        "iterations": current_iter + 1,
    }

def critic_node(state: AgentState):
    print("\n[Node] Ranker/Critic: Evaluating product relevance...")
    return ranker_critic.run(state)

def output_node(state: AgentState):
    print("\n[Node] Output: Formatting results...")
    # Check if we are in a fallback state (reached max iterations without high score)
    ranked = state.get("ranked_products", [])
    max_score = max([p["score"] for p in ranked]) if ranked else 0
    
    results = output.run(state)
    
    # Append warning if λ threshold was never met
    if max_score <= 0.6:
        results["output_text"] = "LOW MATCH CONFIDENCE\n" + results.get("output_text", "")
    
    return results

# ─── Routing Logic ───────────────────────────────────────────────────

def should_retry(state: AgentState) -> Literal["retry", "continue"]:
    ranked_products = state.get("ranked_products", [])
    iterations = state.get("iterations", 0)
    
    # Relevance Threshold λ > 0.6
    max_score = max([p["score"] for p in ranked_products]) if ranked_products else 0
    
    if max_score > 0.6:
        print(f"--- Success: Found product with score {max_score:.2f} > 0.6 ---")
        return "continue"
    
    if iterations < 3:  # Initial (1) + 2 Retries = 3 total attempts
        print(f"--- Retry: Max score {max_score:.2f} <= 0.6. Starting retry {iterations} ---")
        return "retry"
    
    print("--- Fallback: Max retries reached. Using best available results. ---")
    return "continue"

# ─── Graph Construction ──────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    # Define Nodes
    workflow.add_node("stylist", stylist_node)
    workflow.add_node("procurement", procurement_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("output", output_node)

    # Define Workflow
    workflow.set_entry_point("stylist")
    workflow.add_edge("stylist", "procurement")
    workflow.add_edge("procurement", "critic")

    # The Loop logic
    workflow.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "procurement",
            "continue": "output"
        }
    )

    workflow.add_edge("output", END)
    return workflow.compile()

# ─── Main Execution ──────────────────────────────────────────────────

def get_image_paths_from_url(board_url: str, max_images: int = 5) -> list:
    """Helper to fetch images using your existing crawler."""
    results = pinterest_crawler.crawl_pinterest(board_url, num_boards=1, max_images_per_board=max_images)
    if not results:
        return []
    download_folder = pinterest_crawler.download_images(results, base_dir="data")
    paths = []
    for root, _, files in os.walk(download_folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, f))
    return paths

def run_vision_cart(url: str):
    image_paths = get_image_paths_from_url(url)
    if not image_paths:
        print("No images found.")
        return

    # Initial state matching your AgentState TypedDict
    initial_state: AgentState = {
        "vision_board_paths": image_paths,
        "num_products": 5,
        "iterations": 0,
        "stylist_output": {},
        "procurement_queries": [],
        "candidate_products": [],
        "style_profile": "",
        "ranked_products": [],
        "rejected_products": [],
        "output_text": "",
        "output_products": []
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL PIPELINE OUTPUT")
    print("="*50)
    print(final_state["output_text"])

if __name__ == "__main__":
    test_url = "https://www.pinterest.com/aesthetics/spring-wallpapers/"
    try:
        run_vision_cart(test_url)
    except Exception as e:
        print(f"Error in VisionCart pipeline: {e}")
        sys.exit(1)