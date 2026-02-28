from __future__ import annotations

import numpy as np
import torchvision.transforms as T
from PIL import Image, ExifTags

from src.models import ExifData


def load_image(source) -> tuple[Image.Image, np.ndarray, object]:
    """
    source: file path (str) or file-like object (e.g. Streamlit UploadedFile).
    Returns:
        pil_image  — RGB PIL Image at original resolution
        bgr_array  — uint8 numpy array in BGR order (for OpenCV)
        tensor     — float32 (1, 3, H, W) tensor in [0, 1] (for pyiqa)
    """
    pil_image = Image.open(source).convert("RGB")
    rgb_array = np.array(pil_image)
    bgr_array = rgb_array[:, :, ::-1].copy()
    tensor = T.ToTensor()(pil_image).unsqueeze(0)
    return pil_image, bgr_array, tensor


def extract_exif(pil_image: Image.Image) -> ExifData:
    """Pull EXIF tags from a PIL Image. Returns ExifData with None for missing fields."""
    raw  = pil_image._getexif() or {}  # noqa: SLF001
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}

    def to_float(r):
        try:
            return float(r)
        except Exception:
            return None

    shutter = None
    if "ExposureTime" in tags:
        t = to_float(tags["ExposureTime"])
        if t is not None:
            shutter = f"1/{int(round(1 / t))}" if t < 1 else str(t)

    return ExifData(
        camera_make=tags.get("Make"),
        camera_model=tags.get("Model"),
        iso=tags.get("ISOSpeedRatings"),
        shutter_speed=shutter,
        aperture=to_float(tags.get("FNumber")),
        focal_length=to_float(tags.get("FocalLength")),
        lens_model=tags.get("LensModel"),
        image_width=pil_image.width,
        image_height=pil_image.height,
    )
