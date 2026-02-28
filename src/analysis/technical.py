from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import pyiqa  # type: ignore[import-untyped]
import torch
from skimage.restoration import estimate_sigma

from src.models import TechnicalScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model initialisation (once per process)
# ---------------------------------------------------------------------------
_DEVICE = "cpu"


def _load(name: str) -> Any | None:
    try:
        return pyiqa.create_metric(name, device=_DEVICE)
    except Exception as exc:
        logger.warning("Could not load pyiqa metric %r: %s", name, exc)
        return None


_brisque = _load("brisque")
_nima = _load("nima")
_clip_iqa = _load("clipiqa+")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyse(bgr_array: np.ndarray, tensor: torch.Tensor) -> TechnicalScores:
    """
    Run all Layer-1 metrics on a single image.

    Args:
        bgr_array: uint8 numpy array, shape (H, W, 3), BGR channel order.
        tensor:    float32 torch tensor, shape (1, 3, H, W), values in [0, 1].

    Returns:
        TechnicalScores Pydantic instance.
    """
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY).astype(np.float32)

    return TechnicalScores(
        brisque=_score_brisque(tensor),
        nima_aesthetic=_score_nima(tensor),
        clip_iqa=_score_clip_iqa(tensor),
        sharpness_laplacian=_sharpness_global(gray),
        sharpness_regional=_sharpness_regional(gray),
        noise_sigma=_noise(bgr_array),
        **_exposure(gray),
        dynamic_range_stops=_dynamic_range(gray),
        contrast_rms=_contrast_rms(gray),
    )


# ---------------------------------------------------------------------------
# Sub-components — pyiqa (Sub-component A)
# ---------------------------------------------------------------------------
def _score_brisque(tensor: torch.Tensor) -> float:
    if _brisque is None:
        return float("nan")
    try:
        return float(_brisque(tensor).item())
    except Exception as exc:
        logger.warning("brisque failed: %s", exc)
        return float("nan")


def _score_nima(tensor: torch.Tensor) -> float | None:
    if _nima is None:
        return None
    try:
        return float(_nima(tensor).item())
    except Exception as exc:
        logger.warning("nima failed: %s", exc)
        return None


def _score_clip_iqa(tensor: torch.Tensor) -> float | None:
    if _clip_iqa is None:
        return None
    try:
        return float(_clip_iqa(tensor).item())
    except Exception as exc:
        logger.warning("clip_iqa+ failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Sub-components — Classical CV (Sub-component B)
# ---------------------------------------------------------------------------
def _sharpness_global(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _sharpness_regional(gray: np.ndarray) -> dict:
    h, w = gray.shape
    mh, mw = h // 2, w // 2
    quadrants = {
        "top_left": gray[:mh, :mw],
        "top_right": gray[:mh, mw:],
        "bottom_left": gray[mh:, :mw],
        "bottom_right": gray[mh:, mw:],
    }
    return {k: float(cv2.Laplacian(q, cv2.CV_32F).var()) for k, q in quadrants.items()}


def _noise(bgr_array: np.ndarray) -> float:
    rgb = bgr_array[:, :, ::-1]
    sigma = estimate_sigma(rgb, channel_axis=-1, average_sigmas=True)
    return float(sigma)


def _exposure(gray: np.ndarray) -> dict:
    total = gray.size
    highlights_pct = float(np.sum(gray >= 250) / total * 100)
    shadows_pct = float(np.sum(gray <= 5) / total * 100)
    return {
        "exposure_clipped_highlights_pct": highlights_pct,
        "exposure_clipped_shadows_pct": shadows_pct,
        "histogram_mean": float(gray.mean()),
        "histogram_std": float(gray.std()),
    }


def _dynamic_range(gray: np.ndarray) -> float:
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    if p1 <= 0:
        return 0.0
    return float(math.log2(p99 / p1))


def _contrast_rms(gray: np.ndarray) -> float:
    mean = gray.mean()
    if mean == 0:
        return 0.0
    return float(gray.std() / mean)
