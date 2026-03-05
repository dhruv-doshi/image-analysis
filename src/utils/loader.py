from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ExifTags, Image

from src.models import ExifData

_MAX_ANALYSIS_DIM = 1024  # pyiqa ViT models (MUSIQ, CLIP-IQA+) are O(patches²) — cap here


def load_image(
    source: str | Path | IO[bytes],
) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    """
    source: file path (str / Path) or file-like object (e.g. Streamlit UploadedFile).
    Returns:
        pil_image  — RGB PIL Image, resized to max _MAX_ANALYSIS_DIM on the long edge
        bgr_array  — uint8 numpy array in BGR order (for OpenCV)
        tensor     — float32 (1, 3, H, W) tensor in [0, 1] (for pyiqa)

    Images larger than _MAX_ANALYSIS_DIM px on the long edge are downsampled with
    high-quality Lanczos resampling. All metrics are scale-invariant so quality is
    unaffected; this prevents ViT-based models (MUSIQ, CLIP-IQA+) from hanging on
    high-resolution photos.
    """
    img = Image.open(source)
    exif_bytes = img.info.get("exif", b"")
    pil_image = img.convert("RGB")
    if exif_bytes:
        pil_image.info["exif"] = exif_bytes
    # thumbnail() only shrinks — no-op on small images
    if max(pil_image.width, pil_image.height) > _MAX_ANALYSIS_DIM:
        pil_image.thumbnail((_MAX_ANALYSIS_DIM, _MAX_ANALYSIS_DIM), Image.LANCZOS)
    rgb_array = np.array(pil_image)
    bgr_array = rgb_array[:, :, ::-1].copy()
    tensor = transforms.ToTensor()(pil_image).unsqueeze(0)
    return pil_image, bgr_array, tensor


def extract_exif(pil_image: Image.Image) -> ExifData:
    """Pull EXIF tags from a PIL Image. Returns ExifData with None for missing fields."""
    try:
        raw = dict(pil_image.getexif())
    except Exception:
        raw = {}
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}

    def to_float(r: object) -> float | None:
        try:
            return float(r)  # type: ignore[arg-type]
        except Exception:
            return None

    shutter = None
    if "ExposureTime" in tags:
        t = to_float(tags["ExposureTime"])
        if t is not None:
            shutter = f"1/{int(round(1 / t))}" if t < 1 else str(t)

    # Prefer EXIF-embedded original dimensions over PIL's (which may be post-resize)
    orig_w = tags.get("ExifImageWidth") or tags.get("ImageWidth") or pil_image.width
    orig_h = tags.get("ExifImageHeight") or tags.get("ImageLength") or pil_image.height

    return ExifData(
        camera_make=tags.get("Make"),
        camera_model=tags.get("Model"),
        iso=tags.get("ISOSpeedRatings"),
        shutter_speed=shutter,
        aperture=to_float(tags.get("FNumber")),
        focal_length=to_float(tags.get("FocalLength")),
        lens_model=tags.get("LensModel"),
        image_width=int(orig_w) if orig_w is not None else None,
        image_height=int(orig_h) if orig_h is not None else None,
    )
