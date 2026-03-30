import json
import re
import os
import torch
from typing import Dict, Any, List
from PIL import Image
from dotenv import load_dotenv
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig
)

load_dotenv()
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

if not os.environ.get("HF_TOKEN"):
    print("HF_TOKEN not found in environment. Downloads may be slow or fail.")

# 3. Load Model and Processor
# In src/agents/stylist.py (When running on Colab)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto", # Full precision for better "vibe" detection
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stylist Agent: Converts Pinterest images into a Style Persona.
    
    Inputs: 
        state['vision_board_paths']: List of local file paths to images.
    Outputs: 
        Adds 'stylist_output' to the state, matching the Procurement Agent's contract.
    """
    image_paths = state.get("vision_board_paths", [])
    if not image_paths:
        return {"stylist_output": {}}

    # Load images (limit to 5 to save memory/tokens for now)
    images = []
    for path in image_paths[:5]:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    # prompt: Enforcing the exact schema the Procurement Agent expects
    prompt = """Analyze these images and create a 'Style Persona'. 
    Return ONLY a JSON object with these exact keys:
    {
      "style_profile": "A narrative prose describing the overall vibe.",
      "products": ["List of specific item types to search for (e.g. 'planters')"],
      "aesthetic": ["Mood descriptors (e.g. 'cottagecore', 'boho')"],
      "colors": ["3-5 specific colors found in the images"],
      "materials": ["3-5 materials (e.g. 'rattan', 'oak')"],
      "budget_max": 100.0,
      "budget_currency": "USD"
    }"""

    # Prepare Multimodal Input
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Generate Output
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], 
        images=images, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        # 1. Use regex to find the first '{' and last '}' 
        # The re.DOTALL flag ensures it matches across multiple lines
        json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
        
        if json_match:
            clean_json_str = json_match.group(1)
            stylist_json = json.loads(clean_json_str)
        else:
            raise ValueError("No JSON object found in output")

    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        stylist_json = {
            "style_profile": "Unknown",
            "products": [],
            "aesthetic": [],
            "colors": [],
            "materials": []
        }

    return {"stylist_output": stylist_json}