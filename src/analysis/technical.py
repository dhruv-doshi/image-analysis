from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import pyiqa
import torch
from skimage.restoration import estimate_sigma

from src.config.metrics import is_enabled
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
_musiq = _load("musiq")
_niqe = _load("niqe")


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

    exp: dict = {}
    if any(
        is_enabled(k)
        for k in (
            "exposure_clipped_highlights_pct",
            "exposure_clipped_shadows_pct",
            "histogram_mean",
            "histogram_std",
        )
    ):
        exp = _exposure(gray)

    return TechnicalScores(
        brisque=_score_brisque(tensor) if is_enabled("brisque") else None,
        nima_aesthetic=_score_nima(tensor) if is_enabled("nima_aesthetic") else None,
        clip_iqa=_score_clip_iqa(tensor) if is_enabled("clip_iqa") else None,
        musiq=_score_musiq(tensor) if is_enabled("musiq") else None,
        niqe=_score_niqe(tensor) if is_enabled("niqe") else None,
        sharpness_laplacian=_sharpness_global(gray) if is_enabled("sharpness_laplacian") else None,
        sharpness_regional=_sharpness_regional(gray) if is_enabled("sharpness_regional") else {},
        noise_sigma=_noise(bgr_array) if is_enabled("noise_sigma") else None,
        exposure_clipped_highlights_pct=exp.get("exposure_clipped_highlights_pct")
        if is_enabled("exposure_clipped_highlights_pct")
        else None,
        exposure_clipped_shadows_pct=exp.get("exposure_clipped_shadows_pct")
        if is_enabled("exposure_clipped_shadows_pct")
        else None,
        histogram_mean=exp.get("histogram_mean") if is_enabled("histogram_mean") else None,
        histogram_std=exp.get("histogram_std") if is_enabled("histogram_std") else None,
        dynamic_range_stops=_dynamic_range(gray) if is_enabled("dynamic_range_stops") else None,
        contrast_rms=_contrast_rms(gray) if is_enabled("contrast_rms") else None,
    )


# ---------------------------------------------------------------------------
# Sub-components — pyiqa (Sub-component A)
# ---------------------------------------------------------------------------
def _score_brisque(tensor: torch.Tensor) -> float:
    if _brisque is None:
        return float("nan")
    try:
        import time

        t = time.perf_counter()
        val = float(_brisque(tensor).item())
        logger.debug("brisque complete  %.2fs  val=%.1f", time.perf_counter() - t, val)
        return val
    except Exception as exc:
        logger.warning("brisque failed: %s", exc)
        return float("nan")


def _score_nima(tensor: torch.Tensor) -> float | None:
    if _nima is None:
        return None
    try:
        import time

        t = time.perf_counter()
        val = float(_nima(tensor).item())
        logger.debug("nima complete  %.2fs  val=%.1f", time.perf_counter() - t, val)
        return val
    except Exception as exc:
        logger.warning("nima failed: %s", exc)
        return None


def _score_clip_iqa(tensor: torch.Tensor) -> float | None:
    if _clip_iqa is None:
        return None
    try:
        import time

        t = time.perf_counter()
        val = float(_clip_iqa(tensor).item())
        logger.debug("clip_iqa+ complete  %.2fs  val=%.1f", time.perf_counter() - t, val)
        return val
    except Exception as exc:
        logger.warning("clip_iqa+ failed: %s", exc)
        return None


def _score_musiq(tensor: torch.Tensor) -> float | None:
    if _musiq is None:
        return None
    try:
        import time

        t = time.perf_counter()
        val = float(_musiq(tensor).item())
        logger.debug("musiq complete  %.2fs  val=%.1f", time.perf_counter() - t, val)
        return val
    except Exception as exc:
        logger.warning("musiq failed: %s", exc)
        return None


def _score_niqe(tensor: torch.Tensor) -> float | None:
    if _niqe is None:
        return None
    try:
        import time

        t = time.perf_counter()
        val = float(_niqe(tensor).item())
        logger.debug("niqe complete  %.2fs  val=%.1f", time.perf_counter() - t, val)
        return val
    except Exception as exc:
        logger.warning("niqe failed: %s", exc)
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
    return {
        k: float(cv2.Laplacian(q, cv2.CV_32F).var()) if q.size > 0 else 0.0
        for k, q in quadrants.items()
    }


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
