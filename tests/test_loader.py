"""
Unit tests for *src/utils/loader.py* — ``load_image`` and ``extract_exif``.

All tests use synthetic PIL images; no real photo files are needed.
piexif-dependent tests are automatically skipped when piexif is absent.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.models import ExifData
from src.utils.loader import extract_exif, load_image


# ---------------------------------------------------------------------------
# Local helper fixture — avoids repeating the JPEG save/path logic
# ---------------------------------------------------------------------------

@pytest.fixture()
def grey_jpg(solid_grey_image, tmp_path) -> str:
    """Save the solid-grey fixture as a JPEG and return the file path (str)."""
    path = tmp_path / "grey.jpg"
    solid_grey_image.save(str(path), format="JPEG")
    return str(path)


# ===========================================================================
# load_image — from a file path
# ===========================================================================

class TestLoadImageFromPath:

    def test_returns_three_element_tuple(self, grey_jpg):
        result = load_image(grey_jpg)
        assert len(result) == 3

    def test_first_element_is_pil_image(self, grey_jpg):
        pil, _, _ = load_image(grey_jpg)
        assert isinstance(pil, Image.Image)

    def test_pil_mode_is_rgb(self, grey_jpg):
        pil, _, _ = load_image(grey_jpg)
        assert pil.mode == "RGB"

    def test_bgr_is_ndarray(self, grey_jpg):
        _, bgr, _ = load_image(grey_jpg)
        assert isinstance(bgr, np.ndarray)

    def test_bgr_is_three_dimensional(self, grey_jpg):
        _, bgr, _ = load_image(grey_jpg)
        assert bgr.ndim == 3

    def test_bgr_has_three_channels(self, grey_jpg):
        _, bgr, _ = load_image(grey_jpg)
        assert bgr.shape[2] == 3

    def test_bgr_shape_matches_pil_dimensions(self, grey_jpg):
        pil, bgr, _ = load_image(grey_jpg)
        assert bgr.shape == (pil.height, pil.width, 3)

    def test_bgr_dtype_is_uint8(self, grey_jpg):
        _, bgr, _ = load_image(grey_jpg)
        assert bgr.dtype == np.uint8

    def test_tensor_is_torch_tensor(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_has_four_dimensions(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert tensor.ndim == 4

    def test_tensor_batch_size_is_one(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert tensor.shape[0] == 1

    def test_tensor_has_three_channels(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert tensor.shape[1] == 3

    def test_tensor_dtype_is_float32(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert tensor.dtype == torch.float32

    def test_tensor_min_is_non_negative(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert float(tensor.min()) >= 0.0

    def test_tensor_max_is_at_most_one(self, grey_jpg):
        _, _, tensor = load_image(grey_jpg)
        assert float(tensor.max()) <= 1.0


# ===========================================================================
# load_image — from a file-like object (BytesIO)
# ===========================================================================

class TestLoadImageFromBytesIO:

    @pytest.fixture()
    def grey_buf(self, solid_grey_image) -> io.BytesIO:
        buf = io.BytesIO()
        solid_grey_image.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_returns_three_element_tuple(self, grey_buf):
        result = load_image(grey_buf)
        assert len(result) == 3

    def test_first_element_is_pil_image(self, grey_buf):
        pil, _, _ = load_image(grey_buf)
        assert isinstance(pil, Image.Image)

    def test_pil_mode_is_rgb(self, grey_buf):
        pil, _, _ = load_image(grey_buf)
        assert pil.mode == "RGB"

    def test_bgr_is_ndarray(self, grey_buf):
        _, bgr, _ = load_image(grey_buf)
        assert isinstance(bgr, np.ndarray)

    def test_bgr_dtype_is_uint8(self, grey_buf):
        _, bgr, _ = load_image(grey_buf)
        assert bgr.dtype == np.uint8

    def test_tensor_is_torch_tensor(self, grey_buf):
        _, _, tensor = load_image(grey_buf)
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_values_in_unit_interval(self, grey_buf):
        _, _, tensor = load_image(grey_buf)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0


# ===========================================================================
# extract_exif — plain PIL images (no embedded EXIF)
# ===========================================================================

class TestExtractExifNoExif:

    def test_returns_exif_data_instance(self, solid_grey_image):
        result = extract_exif(solid_grey_image)
        assert isinstance(result, ExifData)

    def test_image_width_matches_pil(self, solid_grey_image):
        result = extract_exif(solid_grey_image)
        assert result.image_width == solid_grey_image.width

    def test_image_height_matches_pil(self, solid_grey_image):
        result = extract_exif(solid_grey_image)
        assert result.image_height == solid_grey_image.height

    def test_no_exif_fields_are_none_except_dimensions(self, solid_grey_image):
        """A plain Image.new image has no camera EXIF; only dimensions set."""
        result = extract_exif(solid_grey_image)
        dumped = result.model_dump(exclude_none=True)
        assert set(dumped.keys()) == {"image_width", "image_height"}

    def test_iso_is_none_without_exif(self, solid_grey_image):
        result = extract_exif(solid_grey_image)
        assert result.iso is None

    def test_aperture_is_none_without_exif(self, solid_grey_image):
        result = extract_exif(solid_grey_image)
        assert result.aperture is None


# ===========================================================================
# extract_exif — images with embedded EXIF (requires piexif)
# ===========================================================================

class TestExtractExifWithPiexif:
    """
    All tests in this class are skipped automatically when piexif is not
    installed.  The ``exif_image`` fixture (defined in conftest.py) calls
    ``pytest.importorskip("piexif")`` which handles the skip transparently.
    """

    def test_iso_parsed_correctly(self, exif_image):
        result = extract_exif(exif_image)
        assert result.iso == 400

    def test_aperture_parsed_correctly(self, exif_image):
        result = extract_exif(exif_image)
        assert result.aperture == pytest.approx(2.8, rel=0.02)

    def test_focal_length_parsed_correctly(self, exif_image):
        result = extract_exif(exif_image)
        assert result.focal_length == pytest.approx(50.0, rel=0.02)

    def test_shutter_speed_present(self, exif_image):
        result = extract_exif(exif_image)
        assert result.shutter_speed is not None

    def test_shutter_speed_under_1s_formatted_as_fraction(self, exif_image):
        """1/250 s should be represented as the string '1/250'."""
        result = extract_exif(exif_image)
        assert result.shutter_speed == "1/250"

    def test_shutter_speed_over_1s_formatted_as_decimal(self):
        """3 s should be represented without a fraction (e.g. '3.0')."""
        piexif = pytest.importorskip("piexif")
        img = Image.new("RGB", (10, 10), color=(100, 100, 100))
        exif_dict = {
            "0th": {},
            "Exif": {piexif.ExifIFD.ExposureTime: (3, 1)},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=piexif.dump(exif_dict))
        buf.seek(0)
        pil = Image.open(buf)

        result = extract_exif(pil)

        assert result.shutter_speed is not None
        assert "/" not in result.shutter_speed, (
            "Shutter speed >= 1 s should not be formatted as a fraction"
        )
        assert "3" in result.shutter_speed

    def test_width_height_still_correct_with_exif(self, exif_image):
        result = extract_exif(exif_image)
        assert result.image_width == exif_image.width
        assert result.image_height == exif_image.height


# ===========================================================================
# load_image — resize behaviour (_MAX_ANALYSIS_DIM cap)
# ===========================================================================

class TestLoadImageResize:

    def _large_jpg(self, tmp_path, w: int, h: int) -> str:
        img = Image.new("RGB", (w, h), color=(100, 150, 200))
        path = tmp_path / f"large_{w}x{h}.jpg"
        img.save(str(path), format="JPEG")
        return str(path)

    def test_small_image_not_resized(self, grey_jpg):
        """200×200 image is below the cap — dimensions must be unchanged."""
        pil, bgr, _ = load_image(grey_jpg)
        assert max(pil.width, pil.height) == 200
        assert bgr.shape[:2] == (pil.height, pil.width)

    def test_large_landscape_long_edge_capped(self, tmp_path):
        """2000×1000 image should be downsampled to 1024 on the long edge."""
        from src.utils.loader import _MAX_ANALYSIS_DIM
        path = self._large_jpg(tmp_path, 2000, 1000)
        pil, bgr, tensor = load_image(path)
        assert max(pil.width, pil.height) <= _MAX_ANALYSIS_DIM

    def test_large_portrait_long_edge_capped(self, tmp_path):
        """1000×2000 portrait should be downsampled to 1024 on the long edge."""
        from src.utils.loader import _MAX_ANALYSIS_DIM
        path = self._large_jpg(tmp_path, 1000, 2000)
        pil, bgr, tensor = load_image(path)
        assert max(pil.width, pil.height) <= _MAX_ANALYSIS_DIM

    def test_aspect_ratio_preserved(self, tmp_path):
        """Resizing must not change the aspect ratio beyond rounding error."""
        path = self._large_jpg(tmp_path, 2048, 1024)
        pil, _, _ = load_image(path)
        original_ratio = 2048 / 1024
        resized_ratio = pil.width / pil.height
        assert abs(resized_ratio - original_ratio) < 0.02

    def test_bgr_shape_consistent_with_pil_after_resize(self, tmp_path):
        """BGR array dims must match the (possibly resized) PIL dimensions."""
        path = self._large_jpg(tmp_path, 1600, 1200)
        pil, bgr, _ = load_image(path)
        assert bgr.shape == (pil.height, pil.width, 3)

    def test_tensor_shape_consistent_with_pil_after_resize(self, tmp_path):
        """Tensor H×W must match the resized PIL image."""
        path = self._large_jpg(tmp_path, 1600, 1200)
        pil, _, tensor = load_image(path)
        assert tensor.shape[2] == pil.height
        assert tensor.shape[3] == pil.width
