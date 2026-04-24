import os
import sys
import json
import argparse
from typing import List, Dict, Any
from src.utils import pinterest_crawler
from src.graph.state import AgentState, style_profile_from_stylist_output
from src.agents import stylist, procurement, ranker_critic, output

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
    state["style_profile"] = style_profile_from_stylist_output(state.get("stylist_output") or {})

    print("procurement agent search")
    # returns a JSON string, we must parse it back into the state
    procurement_raw = procurement.run(state)
    procurement_data = json.loads(procurement_raw)
    state["candidate_products"] = procurement_data["candidate_products"]
    state["procurement_queries"] = procurement_data["procurement_queries"]
    state["iterations"] = int(procurement_data.get("iterations", state.get("iterations", 0))) + 1

    if not isinstance(state.get("style_profile"), dict) or not state["style_profile"]:
        raise ValueError("style_profile must be a non-empty dict derived from stylist_output before ranker.")

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
    # parser = argparse.ArgumentParser(description="VisionCart: Pinterest-to-Retail Pipeline")
    # parser.add_argument(
    #     "url", 
    #     help="The Pinterest board URL to analyze"
    # )
    
    # args = parser.parse_args()
    
    # try:
    #     run_vision_cart(args.url)
    # except Exception as e:
    #     print(f"error in pipeline: {e}")
    #     sys.exit(1)

    # For testing purposes, you can hardcode a Pinterest board URL here:
    test_url = "https://www.pinterest.com/aesthetics/spring-wallpapers/"
    try:
        run_vision_cart(test_url, max_images=1)
    except Exception as e:
        print(f"error in pipeline: {e}")
        sys.exit(1)