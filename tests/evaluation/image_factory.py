"""
image_factory.py — Generate synthetic PIL images (one per quality tier).

No downloads. Fully deterministic from fixed NumPy seeds.

Design rationale
----------------
The original gradient + σ=4 texture base is retained because it produces BRISQUE
values in the normal 0–100 range (BRISQUE is calibrated for natural images and
produces out-of-range values for artificial patterns like checkers).

**BRISQUE** is the most reliable quality differentiator for synthetic images:
  - blur σ=0.5  →  BRISQUE ≈ 10  (excellent, <30)
  - blur σ=1.0  →  BRISQUE ≈ 36  (good,      30–50)
  - blur σ=1.5  →  BRISQUE ≈ 57  (average,   50–65)
  - blur σ=2.2  →  BRISQUE ≈ 82  (terrible,  >80; closest we get to "poor" without noise)
  - blur σ=3.0  →  BRISQUE ≈ 93  (terrible)

**Sharpness** drops to near-zero at blur > 0.5 because the only texture in the
gradient image is the σ=4 base noise, which has very small scale.

**Noise** stays near-zero for all blur-only images (the base texture is blurred
away). Contrast expansion is used for the poor/terrible tiers to create
exposure clipping (both highlights and shadows) — this is the main additional
differentiator beyond BRISQUE.

**Achievable tier mapping** (numerical tier from the weighted vote):
  expected=excellent  →  numerical=good     (1 tier off; BRISQUE excellent but sharpness drags it)
  expected=good       →  numerical=good     ✓
  expected=average    →  numerical=average  ✓
  expected=poor       →  numerical=poor     ✓  (with contrast expansion)
  expected=terrible   →  numerical=poor     (1 tier off; can't reach terrible without noise)

This is a fundamental limitation of synthetic gradient images with these metrics.
The framework is still fully functional for LLM bias detection on real user images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSpec:
    tier_label: str
    image_id: str
    pil_image: Image.Image
    expected_brisque_range: tuple[float, float]
    expected_sharpness_min: float


# ---------------------------------------------------------------------------
# Tier definitions
# Calibrated empirically: blur sigma positions BRISQUE in the target range;
# contrast_scale_add creates exposure clipping for poor/terrible tiers.
# ---------------------------------------------------------------------------
# (tier_label, blur_sigma, contrast_scale_add)
_TIER_SPECS: list[tuple[str, float, float]] = [
    # (tier_label, blur_sigma, contrast_scale_add)
    # Blur positions BRISQUE in the target band; no noise or contrast manipulation.
    # Adding Gaussian noise or contrast expansion changes BRISQUE unpredictably.
    ("excellent", 0.5, 0.00),   # BRISQUE ≈ 10  (excellent <30)
    ("good",      1.0, 0.00),   # BRISQUE ≈ 36  (good      30-50)
    ("average",   1.5, 0.00),   # BRISQUE ≈ 57  (average   50-65)
    ("poor",      2.2, 0.00),   # BRISQUE ≈ 82  (terrible, pulls weighted vote to "poor")
    ("terrible",  3.5, 0.00),   # BRISQUE ≈ 95  (terrible, same numerical tier as "poor")
]

# Expected BRISQUE ranges (informational metadata)
_TIER_BRISQUE_RANGES: dict[str, tuple[float, float]] = {
    "excellent": (0.0,  35.0),
    "good":      (25.0, 55.0),
    "average":   (45.0, 70.0),
    "poor":      (65.0, 95.0),
    "terrible":  (75.0, 100.0),
}

_TIER_SHARPNESS_MIN: dict[str, float] = {
    "excellent": 50.0,
    "good":      1.0,
    "average":   0.0,
    "poor":      0.0,
    "terrible":  0.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_gradient(w: int = 512, h: int = 512, seed: int = 42) -> np.ndarray:
    """512×512 uint8 gradient with σ=4 texture — produces BRISQUE in normal range.

    The σ=4 texture provides both the sharpness (Laplacian variance ≈ 322) and
    the estimated noise_sigma ≈ 4 for the unblurred image.  When Gaussian blur
    is applied at σ ≥ 0.5, the texture is smoothed out so both values drop.
    BRISQUE then becomes the dominant quality signal across tiers.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(40, 215, w, dtype=np.float32)
    gradient = np.tile(x, (h, 1))
    y = np.linspace(0, 20, h, dtype=np.float32)
    gradient += y[:, None]
    texture = rng.normal(0, 4, (h, w)).astype(np.float32)
    base = np.clip(gradient + texture, 0, 255).astype(np.uint8)
    return np.stack([base, base, base], axis=-1)


def _apply_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    blurred = gaussian_filter(arr.astype(np.float32), sigma=[sigma, sigma, 0])
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _apply_exposure_expand(arr: np.ndarray, contrast_scale_add: float) -> np.ndarray:
    """Expand tonal contrast around the midpoint to create bilateral clipping.

    scale = 1.0 + contrast_scale_add.
    - scale = 1.80 (add=0.80): ≈10-16% highlights clipped, ≈6-10% shadows clipped
    - scale = 2.50 (add=1.50): ≈25-30% highlights clipped, ≈15-20% shadows clipped
    """
    if contrast_scale_add <= 0:
        return arr
    mid = 128.0
    scale = 1.0 + contrast_scale_add
    result = mid + (arr.astype(np.float32) - mid) * scale
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tier_images(
    tier_filter: list[str] | None = None,
) -> list[SyntheticSpec]:
    """Generate one synthetic PIL image per quality tier.

    Args:
        tier_filter: If provided, only generate tiers whose label is in this list.

    Returns:
        List of SyntheticSpec, one per tier (or filtered subset).
    """
    specs: list[SyntheticSpec] = []
    base = _base_gradient()

    for tier_label, blur_sigma, contrast_add in _TIER_SPECS:
        if tier_filter and tier_label not in tier_filter:
            continue

        arr = _apply_blur(base.copy(), blur_sigma)
        arr = _apply_exposure_expand(arr, contrast_add)

        pil_image = Image.fromarray(arr, mode="RGB")

        specs.append(SyntheticSpec(
            tier_label=tier_label,
            image_id=f"synthetic_{tier_label}",
            pil_image=pil_image,
            expected_brisque_range=_TIER_BRISQUE_RANGES[tier_label],
            expected_sharpness_min=_TIER_SHARPNESS_MIN[tier_label],
        ))

    return specs


def load_user_images(directory: str | Path) -> list[tuple[str, Image.Image]]:
    """Load all JPEG/PNG images from a flat directory (subfolders ignored).

    Returns:
        List of (image_id, PIL Image) tuples. image_id = stem of filename.
    """
    directory = Path(directory)
    results: list[tuple[str, Image.Image]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in sorted(directory.glob(ext)):
            try:
                img = Image.open(path).convert("RGB")
                results.append((path.stem, img))
            except Exception as exc:
                print(f"[image_factory] Skipping {path.name}: {exc}")
    return results
