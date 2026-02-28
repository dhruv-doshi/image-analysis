from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import torch
from PIL import ExifTags, Image
import torchvision.transforms as transforms  # type: ignore[import-untyped]

from src.models import ExifData


def load_image(
    source: str | Path | IO[bytes],
) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    """
    source: file path (str / Path) or file-like object (e.g. Streamlit UploadedFile).
    Returns:
        pil_image  — RGB PIL Image at original resolution
        bgr_array  — uint8 numpy array in BGR order (for OpenCV)
        tensor     — float32 (1, 3, H, W) tensor in [0, 1] (for pyiqa)
    """
    img = Image.open(source)
    exif_bytes = img.info.get("exif", b"")
    pil_image = img.convert("RGB")
    if exif_bytes:
        pil_image.info["exif"] = exif_bytes
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
