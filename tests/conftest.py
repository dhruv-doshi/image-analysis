"""
Shared pytest fixtures for the FrameIQ Layer 1 test suite.

Key design choices
------------------
* pyiqa is mocked at the *sys.modules* level before any src import can
  trigger a real ``pyiqa.create_metric`` call (which would try to download
  model weights).  The stub is installed here, at conftest load time, so it
  is in place during pytest collection.
* The ``_patch_technical_metrics`` autouse fixture then replaces the
  module-level ``_brisque``, ``_nima``, ``_clip_iqa`` callables with
  per-test stubs that return deterministic torch tensors, giving every test
  function a predictable baseline without any model inference.
* All synthetic images are built from numpy arrays / PIL — no real photo
  files are required.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 1.  Mock pyiqa *before* any src.* import happens
# ---------------------------------------------------------------------------

def _iqa_metric_stub(return_value: float) -> MagicMock:
    """Return a callable mock whose result behaves like a 1-element tensor."""
    stub = MagicMock()
    stub.return_value = torch.tensor([[float(return_value)]])
    return stub


def _pyiqa_create_metric(*args, **kwargs) -> MagicMock:
    name = args[0] if args else kwargs.get("metric_name", "")
    return {
        "brisque":  _iqa_metric_stub(50.0),
        "nima":     _iqa_metric_stub(6.0),
        "clipiqa+": _iqa_metric_stub(0.7),
        "musiq":    _iqa_metric_stub(65.0),
    }.get(name, _iqa_metric_stub(0.5))


_PYIQA_STUB = MagicMock()
_PYIQA_STUB.create_metric.side_effect = _pyiqa_create_metric

# setdefault: install only if pyiqa hasn't been imported already
sys.modules.setdefault("pyiqa", _PYIQA_STUB)

# ---------------------------------------------------------------------------
# 2.  Re-usable helper functions (imported by test modules as needed)
# ---------------------------------------------------------------------------

_W, _H = 200, 200  # default synthetic image dimensions (width, height)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a (H, W, 3) uint8 BGR numpy array."""
    import cv2  # noqa: PLC0415 – deferred so conftest loads without cv2 at top
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Return a (1, 3, 224, 224) float32 tensor with values in [0, 1]."""
    from torchvision import transforms  # noqa: PLC0415

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)

# ---------------------------------------------------------------------------
# 3.  Synthetic image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def solid_grey_image() -> Image.Image:
    """200×200 solid grey (128, 128, 128) PIL Image."""
    return Image.new("RGB", (_W, _H), color=(128, 128, 128))


@pytest.fixture()
def all_white_image() -> Image.Image:
    """200×200 all-white PIL Image — exercises highlight clipping."""
    return Image.new("RGB", (_W, _H), color=(255, 255, 255))


@pytest.fixture()
def all_black_image() -> Image.Image:
    """200×200 all-black PIL Image — exercises shadow clipping."""
    return Image.new("RGB", (_W, _H), color=(0, 0, 0))


@pytest.fixture()
def noisy_image() -> Image.Image:
    """200×200 random-noise PIL Image (seeded for reproducibility)."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture()
def sharp_edges_image() -> Image.Image:
    """200×200 black/white checkerboard — maximises Laplacian sharpness."""
    arr = np.zeros((_H, _W, 3), dtype=np.uint8)
    block = 10
    for r in range(0, _H, block):
        for c in range(0, _W, block):
            if ((r // block) + (c // block)) % 2 == 0:
                arr[r : r + block, c : c + block] = 255
    return Image.fromarray(arr, "RGB")


@pytest.fixture()
def exif_image() -> Image.Image:
    """
    100×100 JPEG with synthetic EXIF: ISO 400, f/2.8, 50 mm, 1/250 s.

    Skipped automatically when *piexif* is not installed.
    """
    piexif = pytest.importorskip("piexif")
    img = Image.new("RGB", (100, 100), color=(120, 130, 140))
    exif_dict = {
        "0th": {},
        "Exif": {
            piexif.ExifIFD.ISOSpeedRatings: 400,
            piexif.ExifIFD.FNumber:         (280, 100),   # f/2.8
            piexif.ExifIFD.FocalLength:     (50, 1),      # 50 mm
            piexif.ExifIFD.ExposureTime:    (1, 250),     # 1/250 s
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=piexif.dump(exif_dict))
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# 4.  Pre-computed BGR array and tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def solid_grey_bgr(solid_grey_image):
    return pil_to_bgr(solid_grey_image)

@pytest.fixture()
def solid_grey_tensor(solid_grey_image):
    return pil_to_tensor(solid_grey_image)

@pytest.fixture()
def all_white_bgr(all_white_image):
    return pil_to_bgr(all_white_image)

@pytest.fixture()
def all_white_tensor(all_white_image):
    return pil_to_tensor(all_white_image)

@pytest.fixture()
def all_black_bgr(all_black_image):
    return pil_to_bgr(all_black_image)

@pytest.fixture()
def all_black_tensor(all_black_image):
    return pil_to_tensor(all_black_image)

@pytest.fixture()
def noisy_bgr(noisy_image):
    return pil_to_bgr(noisy_image)

@pytest.fixture()
def noisy_tensor(noisy_image):
    return pil_to_tensor(noisy_image)

@pytest.fixture()
def sharp_edges_bgr(sharp_edges_image):
    return pil_to_bgr(sharp_edges_image)

@pytest.fixture()
def sharp_edges_tensor(sharp_edges_image):
    return pil_to_tensor(sharp_edges_image)

# ---------------------------------------------------------------------------
# 5.  Indirect-parametrization helper fixture
#
#     Usage:
#         @pytest.mark.parametrize("image_arrays", ["solid_grey_image", ...],
#                                  indirect=True)
#         def test_foo(image_arrays):
#             bgr, tensor = image_arrays
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_arrays(request):
    """Return *(bgr_array, tensor)* for the PIL image fixture named by param."""
    pil = request.getfixturevalue(request.param)
    return pil_to_bgr(pil), pil_to_tensor(pil)

# ---------------------------------------------------------------------------
# 6.  autouse: patch module-level pyiqa callables before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_technical_metrics(monkeypatch):
    """
    Replace ``_brisque``, ``_nima``, ``_clip_iqa`` in
    *src.analysis.technical* with deterministic stubs.

    ``raising=False`` means no error if the attribute is missing (e.g. if the
    source module hasn't been written yet or uses a different variable name).
    """
    def _stub(val: float) -> MagicMock:
        m = MagicMock()
        m.return_value = torch.tensor([[float(val)]])
        return m

    try:
        import src.analysis.technical as _tech  # noqa: PLC0415
        monkeypatch.setattr(_tech, "_brisque",  _stub(50.0), raising=False)
        monkeypatch.setattr(_tech, "_nima",     _stub(6.0),  raising=False)
        monkeypatch.setattr(_tech, "_clip_iqa", _stub(0.7),  raising=False)
        monkeypatch.setattr(_tech, "_musiq",    _stub(65.0), raising=False)
    except ImportError:
        # src not on PYTHONPATH yet; loader/model tests still pass fine.
        pass
