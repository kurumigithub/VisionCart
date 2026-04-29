"""CLIP image embedding utilities for VisionCart.

Produces L2-normalised float vectors using openai/clip-vit-base-patch32
(512-dim).  The model is lazy-loaded on first call so importing this module
is safe on CPU-only machines.

Public API
----------
compute_board_embeddings(images_base_dir, style_labels)
    → Dict[style_label, embedding]   (mean of all images in the style folder)

compute_product_embeddings(dataset_path, prefer_local=True)
    → Dict[product_name_lower, embedding]

load_board_embeddings(path)   → Dict loaded from JSON, {} if missing
load_product_embeddings(path) → Dict loaded from JSON, {} if missing
save_embeddings(data, path)   → writes JSON to path
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

BOARD_EMBEDDINGS_PATH   = Path("data/embeddings/board_embeddings.json")
PRODUCT_EMBEDDINGS_PATH = Path("data/embeddings/product_embeddings.json")

_model     = None
_processor = None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clip():
    """Lazy-load CLIP. Safe to import without a GPU."""
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        import torch
        print(f"[clip] Loading {CLIP_MODEL_ID} …")
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("[clip] Model moved to CUDA.")
        else:
            print("[clip] Running on CPU (slower but works).")
    return _model, _processor


# ── Single-image embedding ────────────────────────────────────────────────────

def _embed_pil(img) -> Optional[List[float]]:
    """Embed an already-opened PIL image. Returns None on failure."""
    try:
        import torch
        import torch.nn.functional as F
        model, processor = load_clip()
        inputs = processor(images=img, return_tensors="pt")
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            # Call vision_model + visual_projection explicitly to avoid
            # version-dependent behaviour of get_image_features().
            vision_out = model.vision_model(pixel_values=pixel_values)
            feat = model.visual_projection(vision_out.pooler_output)
            feat = F.normalize(feat, dim=-1)
        return feat[0].cpu().tolist()
    except Exception as e:
        print(f"  [clip] Embedding failed: {e}")
        return None


def embed_image_file(path: str) -> Optional[List[float]]:
    """Embed a local image file. Returns None on failure."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return _embed_pil(img)
    except Exception as e:
        print(f"  [clip] Cannot open {path}: {e}")
        return None


def embed_image_url(url: str) -> Optional[List[float]]:
    """Download and embed an image from a URL. Returns None on failure."""
    try:
        from PIL import Image
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return _embed_pil(img)
    except Exception as e:
        print(f"  [clip] Cannot fetch {url[:70]}: {e}")
        return None


# ── Mean pooling ──────────────────────────────────────────────────────────────

def _mean_embed(vecs: List[List[float]]) -> List[float]:
    """Average a list of L2-normalised embeddings and re-normalise."""
    if not vecs:
        return []
    n, dim = len(vecs), len(vecs[0])
    avg = [sum(vecs[i][d] for i in range(n)) / n for d in range(dim)]
    norm = sum(x * x for x in avg) ** 0.5
    return [x / norm for x in avg] if norm > 0 else avg


# ── Batch helpers ─────────────────────────────────────────────────────────────

def compute_board_embeddings(
    images_base_dir: Path,
    style_labels: List[str],
) -> Dict[str, List[float]]:
    """Compute one mean CLIP embedding per style from vision board images.

    Each style's images live in  <images_base_dir>/<style_label>/*.{jpg,jpeg,png}.
    Styles with no image folder or no images are skipped silently.
    """
    results: Dict[str, List[float]] = {}
    for style in style_labels:
        style_dir = Path(images_base_dir) / style
        if not style_dir.is_dir():
            print(f"  [clip] '{style}': no image dir, skipping.")
            continue
        paths = sorted(
            str(f) for f in style_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            print(f"  [clip] '{style}': empty dir, skipping.")
            continue
        print(f"  [clip] '{style}': embedding {len(paths)} image(s) …")
        vecs = [v for v in (embed_image_file(p) for p in paths) if v is not None]
        if vecs:
            results[style] = _mean_embed(vecs)
            print(f"  [clip] '{style}': done (dim={len(results[style])}).")
    return results


def compute_product_embeddings(
    dataset_path: Path,
    prefer_local: bool = True,
) -> Dict[str, List[float]]:
    """Compute one CLIP embedding per unique product in dataset.json.

    Keyed by product_name.lower() since product_id is assigned at eval time.
    Uses image_local (local file) when prefer_local=True and the file exists,
    otherwise falls back to image_url.
    """
    with open(dataset_path) as f:
        products = json.load(f)

    results: Dict[str, List[float]] = {}
    seen: set = set()
    total = len(products)

    for i, p in enumerate(products, 1):
        name = (p.get("product_name") or "").lower().strip()
        if not name or name in seen:
            continue
        seen.add(name)

        vec: Optional[List[float]] = None

        if prefer_local:
            local = p.get("image_local", "")
            if local and Path(local).exists():
                print(f"  [clip] [{i}/{total}] local  {local}")
                vec = embed_image_file(local)

        if vec is None:
            url = p.get("image_url", "")
            if url:
                print(f"  [clip] [{i}/{total}] url    {url[:70]} …")
                vec = embed_image_url(url)

        if vec is not None:
            results[name] = vec
        else:
            print(f"  [clip] [{i}/{total}] FAILED: {name}")

    return results


# ── Persistence ───────────────────────────────────────────────────────────────

def load_board_embeddings(
    path: Path = BOARD_EMBEDDINGS_PATH,
) -> Dict[str, List[float]]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def load_product_embeddings(
    path: Path = PRODUCT_EMBEDDINGS_PATH,
) -> Dict[str, List[float]]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_embeddings(data: Dict[str, List[float]], path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)
    print(f"[clip] Saved {len(data)} embeddings → {p}")
