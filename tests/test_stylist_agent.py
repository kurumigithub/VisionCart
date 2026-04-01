import os
import json
import pytest
from src.agents import stylist

# mock state
# Replace these paths with 2-3 actual images
MOCK_IMAGE_DIR = "data/test_samples"
os.makedirs(MOCK_IMAGE_DIR, exist_ok=True)

MOCK_STATE = {
    "vision_board_paths": [
        os.path.join(MOCK_IMAGE_DIR, f) for f in os.listdir(MOCK_IMAGE_DIR) 
        if f.endswith(('.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG'))
    ],
    "num_products": 5
}

def test_stylist_output_structure():
    """
    Validates that the Stylist Agent produces the exact keys 
    the Procurement Agent expects.
    """
    print("\nStarting Stylist Agent Test...")
    
    if not MOCK_STATE["vision_board_paths"]:
        pytest.fail(f"No images found in {MOCK_IMAGE_DIR}. Please add sample images to test.")

    # run the agent logic
    updated_state = stylist.run(MOCK_STATE)
    
    # extract the dictionary for procurement agent
    output = updated_state.get("stylist_output", {})
    
    # validation
    required_keys = ["style_profile", "products", "aesthetic", "colors", "materials"]
    
    print("\n--- Agent Output ---")
    print(json.dumps(output, indent=2))
    
    for key in required_keys:
        assert key in output, f"Missing required key: {key}"
        assert output[key], f"Key {key} is empty"
        
    print("\nStylist output is compatible with Procurement Agent.")

if __name__ == "__main__":
    try:
        test_stylist_output_structure()
    except Exception as e:
        print(f"\nTest Failed: {e}")