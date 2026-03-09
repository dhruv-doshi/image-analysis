"""Layer 0: Monocular depth estimation using MiDaS Small (via torch.hub).

Uses the same PyTorch that's already installed by pyiqa — no extra heavyweight
dependencies needed.  MiDaS Small (~90 MB, downloaded once to torch hub cache)
outputs an inverse-depth / disparity map: higher values mean the surface is
closer to the camera, which maps naturally to a Z-axis pop-out in Three.js.
"""
from __future__ import annotations

import base64
import io
import logging

import numpy as np
import torch
import torch.nn.functional as tf_functional
from PIL import Image

logger = logging.getLogger("frameiq.depth")

# Reduce long-edge to this many pixels before depth inference.
# 256 px → ≤65 536 vertices in the Three.js point cloud — very fast to render.
_MAX_DIM = 256

_model = None
_transform = None


def _get_model():
    global _model, _transform  # noqa: PLW0603
    if _model is None:
        logger.info("Loading MiDaS Small model (first call downloads ~90 MB to torch hub cache)…")
        _model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
            verbose=False,
        )
        _model.eval()
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
            verbose=False,
        )
        _transform = midas_transforms.small_transform
        logger.info("MiDaS model ready")
    return _model, _transform


def estimate_depth(pil_image: Image.Image) -> dict:
    """Run monocular depth estimation on *pil_image*.

    Returns a dict with:
      width, height  — dimensions of the downscaled image
      depth_map      — base64-encoded grayscale PNG (disparity: brighter = closer)
      image          — base64-encoded RGB JPEG at the same resolution
    """
    # --- Downscale -------------------------------------------------------
    w, h = pil_image.size
    scale = min(_MAX_DIM / w, _MAX_DIM / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_small = pil_image.resize((new_w, new_h), Image.LANCZOS).convert("RGB")

    # --- Depth inference -------------------------------------------------
    model, transform = _get_model()

    # MiDaS expects a numpy uint8 RGB array
    img_np = np.array(img_small)
    input_batch = transform(img_np)

    with torch.no_grad():
        prediction = model(input_batch)
        # Resize output to match the colour image dimensions
        prediction = tf_functional.interpolate(
            prediction.unsqueeze(1),
            size=[new_h, new_w],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_np = prediction.cpu().numpy().astype(np.float32)

    # --- Normalise to 0-255 for PNG encoding ----------------------------
    d_min, d_max = float(depth_np.min()), float(depth_np.max())
    if d_max > d_min:
        depth_np = (depth_np - d_min) / (d_max - d_min) * 255
    depth_pil = Image.fromarray(depth_np.astype(np.uint8), mode="L")

    # --- Encode ----------------------------------------------------------
    buf_depth = io.BytesIO()
    depth_pil.save(buf_depth, format="PNG")
    depth_b64 = base64.b64encode(buf_depth.getvalue()).decode()

    buf_img = io.BytesIO()
    img_small.save(buf_img, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf_img.getvalue()).decode()

    logger.info("Depth estimation complete  size=%dx%d", new_w, new_h)
    return {
        "width": new_w,
        "height": new_h,
        "depth_map": depth_b64,
        "image": image_b64,
    }
