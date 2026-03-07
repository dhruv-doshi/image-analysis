#!/usr/bin/env python3
"""Generate test fixture images used by Playwright e2e tests and pytest.

Creates:
  tests/fixtures/photo.jpg  — 400x300 gradient+noise image (passes pre-screening)
  tests/fixtures/blank.jpg  — 400x300 solid white image   (triggers 422 rejection)

Run from the repo root:
    python scripts/make_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def make_photo(path: Path, width: int = 400, height: int = 300) -> None:
    """Gradient + noise — high std, passes Tier-1 pre-screen."""
    rng = np.random.default_rng(seed=0)
    # Horizontal gradient base
    grad = np.linspace(40, 220, width, dtype=np.float32)
    arr = np.tile(grad, (height, 1))
    # Vertical gradient modulation
    vgrad = np.linspace(0.7, 1.3, height, dtype=np.float32)[:, np.newaxis]
    arr = np.clip(arr * vgrad, 0, 255).astype(np.uint8)
    # Add colour channels with slight offsets and noise
    r = np.clip(arr.astype(np.float32) + rng.normal(0, 12, arr.shape), 0, 255).astype(np.uint8)
    g = np.clip(arr.astype(np.float32) + rng.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
    b = np.clip(arr.astype(np.float32) + rng.normal(0, 14, arr.shape), 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    Image.fromarray(rgb, "RGB").save(path, "JPEG", quality=90)
    print(f"  photo.jpg  std={rgb.std():.1f}  → {path}")


def make_blank(path: Path, width: int = 400, height: int = 300) -> None:
    """Solid white — std=0, rejected by pre-screening Tier 1."""
    Image.new("RGB", (width, height), (255, 255, 255)).save(path, "JPEG", quality=90)
    print(f"  blank.jpg  std=0.0  → {path}")


if __name__ == "__main__":
    print("Generating test fixtures …")
    make_photo(FIXTURES / "photo.jpg")
    make_blank(FIXTURES / "blank.jpg")
    print("Done.")
