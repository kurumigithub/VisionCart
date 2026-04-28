import os
import sys
import json
import argparse
from typing import List, Dict, Any
from src.utils import pinterest_crawler
from src.graph.state import AgentState
from src.agents import stylist, procurement, ranker_critic, output


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


def get_image_paths_from_url(board_url: str, max_images: int = 5) -> List[str]:
    """
    Crawls a Pinterest board, downloads images to the data/ folder,
    and returns a list of local file paths.
    """
    print(f"crawling pinterest board: {board_url}")
    
    # fetch board info and image URLs
    results = pinterest_crawler.crawl_pinterest(
        board_url, 
        num_boards=1, 
        max_images_per_board=max_images
    )
    
    if not results:
        print("No images found on the provided board URL.")
        return []

    # download images to data/YYYY-MM-DD_HH-MM-SS/
    download_folder = pinterest_crawler.download_images(results, base_dir="data")
    print(f"images downloaded to: {download_folder}")

    # collect the local paths of the downloaded files
    local_paths = []
    for root, _, files in os.walk(download_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                local_paths.append(os.path.join(root, file))
    
    return local_paths

def run_vision_cart(board_url: str, max_images: int = 5):

    image_paths = get_image_paths_from_url(board_url, max_images)
    
    if not image_paths:
        print("Aborting: No images to process.")
        return
    
    # initial State
    state: AgentState = {
        "vision_board_paths": image_paths,
        "num_products": 5,
        "iterations": 0,
        "stylist_output": {},
        "candidate_products": [],
        "ranked_products": [],
        "output_text": ""
    }

    print("stylist agent analysis")
    state.update(stylist.run(state)) 

    print("procurement agent search")
    # returns a JSON string, we must parse it back into the state
    procurement_raw = procurement.run(state)
    procurement_data = json.loads(procurement_raw)
    state["candidate_products"] = procurement_data["procurement_products"]
    state["procurement_queries"] = procurement_data["procurement_queries"]
    state["style_profile"] = _style_profile_from_stylist(state["stylist_output"])

    print("ranker/critic evaluation")
    ranker_results = ranker_critic.run(state)
    state.update(ranker_results)

    print("output generation")
    output_results = output.run(state)
    state.update(output_results)

    print("\ncomplete analysis")
    print(f"Style Identified: {state['style_profile']}")
    print(f"Results: {len(state['ranked_products'])} products passed the critic.")
    print("-" * 30)
    print(state["output_text"])

    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionCart: Pinterest-to-Retail Pipeline")
    parser.add_argument(
        "url",
        help="The Pinterest board URL to analyze",
    )
    parser.add_argument(
        "--max-images", type=int, default=5,
        help="Maximum images to download from the board (default: 5)",
    )
    args = parser.parse_args()
    try:
        run_vision_cart(args.url, args.max_images)
    except Exception as e:
        print(f"error in pipeline: {e}")
        sys.exit(1)