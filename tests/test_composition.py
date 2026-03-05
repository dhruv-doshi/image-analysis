"""
Unit tests for src/analysis/composition.py — Layer 2.

rembg is mocked at the sys.modules level (below) so it is never actually
called.  All imports of src.analysis.composition happen *inside* test bodies
so this file can be collected even before composition.py is written — in that
case every test will show as ERROR (ImportError), which is the expected TDD
state.

Organisation
------------
TestOutputType        – analyse() → CompositionScores instance
TestCentroid          – _centroid() normalised (x, y) for centre / top-left
TestRotAlignment      – _rot_alignment() near-zero at RoT intersection (1/3, 1/3)
TestNegativeSpace     – _negative_space() for fully-salient / fully-transparent
TestSymmetry          – _symmetry() returns 1.0 for a solid-grey (symmetric) image
TestLeadingLines      – _leading_lines() blank vs checkerboard
test_analyse_smoke    – parametrised end-to-end smoke for all mask fixtures
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# 1.  Mock rembg *before* any src.* import happens
# ---------------------------------------------------------------------------

_rembg_mock = MagicMock()
sys.modules.setdefault("rembg", _rembg_mock)

# ---------------------------------------------------------------------------
# 2.  Module-level helper
# ---------------------------------------------------------------------------


def make_rgba_from_alpha(alpha_array: np.ndarray) -> Image.Image:
    """Build an RGBA PIL Image from a 2-D uint8 alpha array.

    RGB channels are set to white (255); only the alpha channel encodes
    saliency so that composition.py can extract it via image.split()[-1]
    or np.array(image)[:, :, 3].
    """
    h, w = alpha_array.shape
    rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
    rgba[:, :, 3] = alpha_array
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# 3.  Local fixtures
# ---------------------------------------------------------------------------

_W, _H = 200, 200  # default synthetic image dimensions (width × height)


@pytest.fixture()
def centre_mask() -> Image.Image:
    """RGBA image with a filled circle at image centre — centroid ≈ (0.5, 0.5)."""
    import cv2  # noqa: PLC0415

    alpha = np.zeros((_H, _W), dtype=np.uint8)
    cv2.circle(alpha, (_W // 2, _H // 2), radius=30, color=255, thickness=-1)
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def top_left_mask() -> Image.Image:
    """RGBA image with a filled circle in the top-left — centroid in upper-left quadrant."""
    import cv2  # noqa: PLC0415

    alpha = np.zeros((_H, _W), dtype=np.uint8)
    cv2.circle(alpha, (30, 30), radius=20, color=255, thickness=-1)
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def rot_mask() -> Image.Image:
    """
    RGBA image with a tiny circle whose centroid lands on the top-left
    rule-of-thirds intersection (W/3, H/3) — RoT alignment score should be ≈ 0.
    """
    import cv2  # noqa: PLC0415

    alpha = np.zeros((_H, _W), dtype=np.uint8)
    cx = _W // 3  # ≈ 66 → normalised ≈ 0.333
    cy = _H // 3  # ≈ 66 → normalised ≈ 0.333
    cv2.circle(alpha, (cx, cy), radius=5, color=255, thickness=-1)
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def uniform_white() -> Image.Image:
    """Fully-opaque (alpha=255) RGBA image — negative space ratio should be ~0."""
    alpha = np.full((_H, _W), 255, dtype=np.uint8)
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def uniform_black() -> Image.Image:
    """Fully-transparent (alpha=0) RGBA image — negative space ratio should be ~1."""
    alpha = np.zeros((_H, _W), dtype=np.uint8)
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def lr_symmetric_mask() -> Image.Image:
    """
    RGBA image with a left-right symmetric vertical band (columns 75-125,
    full height).  Used as the rembg return value in the smoke test.
    """
    alpha = np.zeros((_H, _W), dtype=np.uint8)
    alpha[:, 75:125] = 255  # centred band — symmetric about vertical axis
    return make_rgba_from_alpha(alpha)


@pytest.fixture()
def blank_bgr() -> np.ndarray:
    """200×200 solid mid-grey BGR image — no edges, no lines expected."""
    return np.full((_H, _W, 3), 128, dtype=np.uint8)


@pytest.fixture()
def checkerboard_bgr() -> np.ndarray:
    """200×200 black/white 10-px checkerboard BGR — maximal edges for line detection."""
    arr = np.zeros((_H, _W, 3), dtype=np.uint8)
    block = 10
    for r in range(0, _H, block):
        for c in range(0, _W, block):
            if ((r // block) + (c // block)) % 2 == 0:
                arr[r : r + block, c : c + block] = 255
    return arr


# ---------------------------------------------------------------------------
# 4.  Test classes
# ---------------------------------------------------------------------------


class TestOutputType:
    """analyse() must return a CompositionScores instance."""

    def test_output_type(self, centre_mask, blank_bgr):
        import src.analysis.composition as comp
        from src.models import CompositionScores

        sys.modules["rembg"].remove.return_value = centre_mask
        pil = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = comp.analyse(blank_bgr, pil)
        assert isinstance(result, CompositionScores)


class TestCentroid:
    """_centroid() returns a normalised (cx, cy) centroid of the alpha mask."""

    def test_centre_mask_centroid_near_half(self, centre_mask):
        import src.analysis.composition as comp

        alpha = np.array(centre_mask)[:, :, 3]
        cx, cy = comp._centroid(alpha)
        assert cx == pytest.approx(0.5, abs=0.05), (
            f"Centre-mask centroid x should be ≈ 0.5, got {cx:.4f}"
        )
        assert cy == pytest.approx(0.5, abs=0.05), (
            f"Centre-mask centroid y should be ≈ 0.5, got {cy:.4f}"
        )

    def test_top_left_mask_centroid_in_upper_left(self, top_left_mask):
        import src.analysis.composition as comp

        alpha = np.array(top_left_mask)[:, :, 3]
        cx, cy = comp._centroid(alpha)
        assert cx < 0.4, (
            f"Top-left centroid x should be < 0.4, got {cx:.4f}"
        )
        assert cy < 0.4, (
            f"Top-left centroid y should be < 0.4, got {cy:.4f}"
        )


class TestRotAlignment:
    """_rot_alignment() returns ≈ 0 when centroid is at a RoT intersection."""

    def test_rot_intersection_score_near_zero(self, rot_mask):
        import src.analysis.composition as comp

        alpha = np.array(rot_mask)[:, :, 3]
        cx, cy = comp._centroid(alpha)
        score = comp._rot_alignment(cx, cy)
        assert score == pytest.approx(0.0, abs=0.1), (
            f"Centroid at ({cx:.3f}, {cy:.3f}) should yield near-zero "
            f"RoT alignment score; got {score:.4f}"
        )


class TestNegativeSpace:
    """_negative_space() measures the fraction of low-alpha (non-salient) pixels."""

    def test_fully_salient_image_has_near_zero_negative_space(self, uniform_white):
        import src.analysis.composition as comp

        alpha = np.array(uniform_white)[:, :, 3]
        ratio = comp._negative_space(alpha)
        assert ratio == pytest.approx(0.0, abs=0.02), (
            f"Fully-opaque image should have ~0 negative space; got {ratio:.4f}"
        )

    def test_fully_transparent_image_has_near_one_negative_space(self, uniform_black):
        import src.analysis.composition as comp

        alpha = np.array(uniform_black)[:, :, 3]
        ratio = comp._negative_space(alpha)
        assert ratio == pytest.approx(1.0, abs=0.02), (
            f"Fully-transparent image should have ~1 negative space; got {ratio:.4f}"
        )


class TestSymmetry:
    """_symmetry() returns NCC-based horizontal/vertical symmetry scores (0–1)."""

    def test_solid_grey_is_perfectly_symmetric(self, blank_bgr):
        """A solid-grey image is trivially LR- and TB-symmetric."""
        import cv2  # noqa: PLC0415
        import src.analysis.composition as comp

        gray = cv2.cvtColor(blank_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h_sym, v_sym = comp._symmetry(gray)
        assert h_sym == pytest.approx(1.0, abs=0.05), (
            f"Solid-grey image should have horizontal symmetry ≈ 1.0; got {h_sym:.4f}"
        )
        assert v_sym == pytest.approx(1.0, abs=0.05), (
            f"Solid-grey image should have vertical symmetry ≈ 1.0; got {v_sym:.4f}"
        )

    def test_symmetry_scores_are_bounded(self, blank_bgr):
        import cv2  # noqa: PLC0415
        import src.analysis.composition as comp

        gray = cv2.cvtColor(blank_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h_sym, v_sym = comp._symmetry(gray)
        assert 0.0 <= h_sym <= 1.0
        assert 0.0 <= v_sym <= 1.0


class TestLeadingLines:
    """_leading_lines() returns (angles: list[float], converges: bool, pattern: str)."""

    def test_blank_image_no_dominant_lines(self, blank_bgr):
        import src.analysis.composition as comp

        angles, converges, pattern = comp._leading_lines(blank_bgr, 0.5, 0.5)
        # Solid-grey has no edges; we expect either no angles or "none" pattern
        assert pattern == "none" or len(angles) == 0, (
            f"Solid-grey image should yield no meaningful lines; "
            f"got pattern={pattern!r}, angles={angles}"
        )

    def test_checkerboard_returns_correct_types(self, checkerboard_bgr):
        import src.analysis.composition as comp

        angles, converges, pattern = comp._leading_lines(checkerboard_bgr, 0.5, 0.5)
        assert isinstance(angles, list), "angles must be a list"
        assert all(isinstance(a, float) for a in angles), (
            "every angle must be a float"
        )
        assert isinstance(converges, bool), "converges must be bool"
        assert isinstance(pattern, str), "pattern must be str"
        assert pattern in {"diagonal", "horizontal", "vertical", "mixed", "none"}, (
            f"pattern must be one of the valid values; got {pattern!r}"
        )


# ---------------------------------------------------------------------------
# 5.  Parametrised smoke test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5b.  New tests for A2/A3/A4
# ---------------------------------------------------------------------------


class TestHorizonTilt:
    """_horizon_tilt() returns signed tilt or None."""

    def test_horizon_tilt_near_zero_for_level_angles(self):
        import src.analysis.composition as comp

        # Near-horizontal angles (< 20° → clockwise near-level)
        angles = [1.0, 2.0, 0.5, 1.5]
        tilt = comp._horizon_tilt(angles)
        assert tilt is not None
        assert abs(tilt) < 5.0, f"Expected near-zero tilt, got {tilt}"

    def test_horizon_tilt_none_on_blank(self, blank_bgr):
        import src.analysis.composition as comp

        angles = comp._detect_lines(blank_bgr)  # solid grey → no lines
        tilt = comp._horizon_tilt(angles)
        assert tilt is None, f"Expected None for blank image, got {tilt}"

    def test_horizon_tilt_none_when_fewer_than_two_lines(self):
        import src.analysis.composition as comp

        # Only one qualifying line → should return None
        tilt = comp._horizon_tilt([5.0])
        assert tilt is None

    def test_horizon_tilt_negative_for_counterclockwise(self):
        import src.analysis.composition as comp

        # Angles near 180° → counterclockwise (negative tilt)
        angles = [170.0, 172.0, 168.0]
        tilt = comp._horizon_tilt(angles)
        assert tilt is not None
        assert tilt < 0, f"Expected negative tilt for 170°+ angles, got {tilt}"


class TestSceneClassification:
    """_classify_scene() returns a valid genre string."""

    def test_scene_smoke_grey_returns_general(self, blank_bgr):
        import src.analysis.composition as comp
        from PIL import Image

        pil = Image.fromarray(np.zeros((blank_bgr.shape[0], blank_bgr.shape[1], 3), dtype=np.uint8))
        scene, _ = comp._classify_scene(blank_bgr, pil, [], None)
        assert scene == "general", f"Solid grey should return 'general', got {scene!r}"

    def test_scene_type_in_valid_set(self, blank_bgr):
        import src.analysis.composition as comp
        from PIL import Image

        pil = Image.fromarray(np.zeros((blank_bgr.shape[0], blank_bgr.shape[1], 3), dtype=np.uint8))
        scene, _ = comp._classify_scene(blank_bgr, pil, [], None)
        assert scene in {"portrait", "landscape", "architecture", "macro", "general"}

    def test_analyse_smoke_includes_scene_type(self, blank_bgr):
        import src.analysis.composition as comp
        from src.models import CompositionScores

        sys.modules["rembg"].remove.return_value = sys.modules["rembg"].remove.return_value
        pil = Image.fromarray(np.zeros((blank_bgr.shape[0], blank_bgr.shape[1], 3), dtype=np.uint8))
        alpha = np.full((blank_bgr.shape[0], blank_bgr.shape[1]), 255, dtype=np.uint8)
        rgba = np.ones((blank_bgr.shape[0], blank_bgr.shape[1], 4), dtype=np.uint8) * 255
        rgba[:, :, 3] = alpha
        sys.modules["rembg"].remove.return_value = Image.fromarray(rgba, "RGBA")
        result = comp.analyse(blank_bgr, pil)
        assert isinstance(result, CompositionScores)
        assert result.scene_type in {"portrait", "landscape", "architecture", "macro", "general"}


class TestColorHarmony:
    """_color_harmony() returns valid harmony type and score."""

    _VALID_TYPES = {
        "monochromatic", "analogous", "complementary",
        "triadic", "split-complementary", "complex",
    }

    def test_color_harmony_monochromatic_on_grey(self, blank_bgr):
        import src.analysis.composition as comp

        _, harmony_type, score = comp._color_harmony(blank_bgr)
        assert harmony_type == "monochromatic", (
            f"Solid grey should be monochromatic, got {harmony_type!r}"
        )
        assert score == pytest.approx(1.0), (
            f"Monochromatic score should be 1.0, got {score}"
        )

    def test_color_harmony_returns_valid_type(self, blank_bgr):
        import src.analysis.composition as comp

        _, harmony_type, score = comp._color_harmony(blank_bgr)
        assert harmony_type in self._VALID_TYPES, (
            f"harmony_type {harmony_type!r} not in valid set"
        )

    def test_color_harmony_score_in_range(self, blank_bgr):
        import src.analysis.composition as comp

        _, _, score = comp._color_harmony(blank_bgr)
        assert 0.0 <= score <= 1.0, f"score {score} out of [0,1] range"

    def test_color_harmony_dominant_colors_length(self, blank_bgr):
        import src.analysis.composition as comp

        colors, _, _ = comp._color_harmony(blank_bgr)
        assert len(colors) == 5, f"Expected 5 dominant colors, got {len(colors)}"


@pytest.mark.parametrize(
    "mask_fixture,bgr_fixture",
    [
        ("centre_mask",       "blank_bgr"),
        ("top_left_mask",     "blank_bgr"),
        ("rot_mask",          "blank_bgr"),
        ("uniform_white",     "blank_bgr"),
        ("uniform_black",     "blank_bgr"),
        ("lr_symmetric_mask", "blank_bgr"),
    ],
    ids=["centre", "top_left", "rot", "white", "black", "lr_sym"],
)
def test_analyse_smoke(mask_fixture, bgr_fixture, request):
    """
    End-to-end smoke: analyse() must return a valid CompositionScores for every
    synthetic fixture without raising any exception, and satisfy structural
    invariants.
    """
    import src.analysis.composition as comp
    from src.models import CompositionScores

    mask = request.getfixturevalue(mask_fixture)
    bgr  = request.getfixturevalue(bgr_fixture)

    sys.modules["rembg"].remove.return_value = mask
    pil = Image.fromarray(np.zeros((bgr.shape[0], bgr.shape[1], 3), dtype=np.uint8))
    result = comp.analyse(bgr, pil)

    assert isinstance(result, CompositionScores)
    # Structural invariants — verify shape and valid ranges
    assert 0.0 <= result.saliency_centroid_x <= 1.0
    assert 0.0 <= result.saliency_centroid_y <= 1.0
    assert 0.0 <= result.rot_alignment_score <= 1.0
    assert 0.0 <= result.golden_ratio_alignment_score <= 1.0
    assert result.best_alignment in {"rule_of_thirds", "golden_ratio"}
    assert 0.0 <= result.negative_space_ratio <= 1.0
    assert set(result.visual_weight_quadrants.keys()) == {
        "top_left", "top_right", "bottom_left", "bottom_right"
    }
    assert result.visual_weight_balance >= 1.0  # heaviest / lightest ≥ 1
    assert 0.0 <= result.symmetry_horizontal <= 1.0
    assert 0.0 <= result.symmetry_vertical <= 1.0
    assert isinstance(result.dominant_line_angles, list)
    assert isinstance(result.leading_lines_converge_to_subject, bool)
    assert result.line_pattern in {"diagonal", "horizontal", "vertical", "mixed", "none"}
    assert result.horizon_tilt_degrees is None or isinstance(result.horizon_tilt_degrees, float)
    assert result.scene_type in {"portrait", "landscape", "architecture", "macro", "general"}
    assert result.color_harmony_type in {
        "monochromatic", "analogous", "complementary",
        "triadic", "split-complementary", "complex",
    }
    assert 0.0 <= result.color_harmony_score <= 1.0
