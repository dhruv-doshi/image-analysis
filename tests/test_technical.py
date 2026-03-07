"""
Unit and integration tests for *src/analysis/technical.py* — Layer 1.

pyiqa metric calls are fully mocked via the ``_patch_technical_metrics``
autouse fixture in conftest.py, so every test runs offline without
downloading model weights.  Classical CV operations (sharpness, noise,
exposure, histogram, dynamic range, contrast) execute for real against
synthetic numpy arrays.

Organisation
------------
TestAnalyseOutputShape   – return type and field presence
TestSharpnessRegional    – four-quadrant regional sharpness structure
TestExposureClipping     – highlight / shadow clipping percentages
TestSharpnessLaplacian   – Laplacian scores relative to image content
TestNoiseSigma           – noise estimation ordering
TestHistogram            – histogram_mean values
TestDynamicRange         – dynamic_range_stops behaviour
TestContrastRms          – contrast_rms behaviour
TestEdgeCases            – ZeroDivisionError guards, single-pixel images
test_smoke_analyse       – parametrised end-to-end smoke test
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.technical import _is_photograph, analyse
from src.models import TechnicalScores

from tests.conftest import pil_to_bgr, pil_to_tensor

# ===========================================================================
# TestAnalyseOutputShape
# ===========================================================================

class TestAnalyseOutputShape:

    def test_returns_technical_scores_instance(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result, TechnicalScores)

    def test_brisque_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.brisque, float)

    def test_sharpness_laplacian_is_float(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.sharpness_laplacian, float)

    def test_noise_sigma_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.noise_sigma, float)

    def test_exposure_highlights_is_float(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.exposure_clipped_highlights_pct, float)

    def test_exposure_shadows_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.exposure_clipped_shadows_pct, float)

    def test_histogram_mean_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.histogram_mean, float)

    def test_histogram_std_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.histogram_std, float)

    def test_dynamic_range_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.dynamic_range_stops, float)

    def test_contrast_rms_is_float(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result.contrast_rms, float)

    def test_nima_aesthetic_is_float_or_none(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.nima_aesthetic is None or isinstance(
            result.nima_aesthetic, float
        )

    def test_clip_iqa_is_float_or_none(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.clip_iqa is None or isinstance(result.clip_iqa, float)


# ===========================================================================
# TestSharpnessRegional
# ===========================================================================

class TestSharpnessRegional:

    def test_regional_dict_has_exactly_four_keys(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert set(result.sharpness_regional.keys()) == {
            "top_left", "top_right", "bottom_left", "bottom_right"
        }

    def test_all_regional_values_are_floats(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        for region, val in result.sharpness_regional.items():
            assert isinstance(val, float), (
                f"sharpness_regional[{region!r}] is {type(val)}, expected float"
            )

    def test_all_regional_values_non_negative(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        for region, val in result.sharpness_regional.items():
            assert val >= 0.0, (
                f"sharpness_regional[{region!r}] = {val} is negative"
            )

    def test_global_sharpness_is_non_negative(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.sharpness_laplacian >= 0.0


# ===========================================================================
# TestExposureClipping
# ===========================================================================

class TestExposureClipping:

    def test_all_white_highlights_heavily_clipped(
        self, all_white_bgr, all_white_tensor
    ):
        result = analyse(all_white_bgr, all_white_tensor)
        assert result.exposure_clipped_highlights_pct > 90.0, (
            f"All-white image should have > 90 % clipped highlights, "
            f"got {result.exposure_clipped_highlights_pct:.1f} %"
        )

    def test_all_white_shadows_not_clipped(
        self, all_white_bgr, all_white_tensor
    ):
        result = analyse(all_white_bgr, all_white_tensor)
        assert result.exposure_clipped_shadows_pct < 5.0, (
            f"All-white image should have < 5 % clipped shadows, "
            f"got {result.exposure_clipped_shadows_pct:.1f} %"
        )

    def test_all_black_shadows_heavily_clipped(
        self, all_black_bgr, all_black_tensor
    ):
        result = analyse(all_black_bgr, all_black_tensor)
        assert result.exposure_clipped_shadows_pct > 90.0, (
            f"All-black image should have > 90 % clipped shadows, "
            f"got {result.exposure_clipped_shadows_pct:.1f} %"
        )

    def test_all_black_highlights_not_clipped(
        self, all_black_bgr, all_black_tensor
    ):
        result = analyse(all_black_bgr, all_black_tensor)
        assert result.exposure_clipped_highlights_pct < 5.0, (
            f"All-black image should have < 5 % clipped highlights, "
            f"got {result.exposure_clipped_highlights_pct:.1f} %"
        )


# ===========================================================================
# TestSharpnessLaplacian
# ===========================================================================

class TestSharpnessLaplacian:

    def test_checkerboard_sharper_than_solid_grey(
        self,
        solid_grey_bgr,
        solid_grey_tensor,
        sharp_edges_bgr,
        sharp_edges_tensor,
    ):
        grey_result  = analyse(solid_grey_bgr,  solid_grey_tensor)
        sharp_result = analyse(sharp_edges_bgr, sharp_edges_tensor)
        assert sharp_result.sharpness_laplacian > grey_result.sharpness_laplacian, (
            "Checkerboard (high-frequency) should score higher than solid grey"
        )


# ===========================================================================
# TestNoiseSigma
# ===========================================================================

class TestNoiseSigma:

    def test_solid_grey_noise_is_very_low(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.noise_sigma < 5.0, (
            f"Solid-grey image should have near-zero noise, "
            f"got noise_sigma = {result.noise_sigma:.2f}"
        )

    def test_noisy_image_has_higher_noise_than_grey(
        self,
        solid_grey_bgr,
        solid_grey_tensor,
        noisy_bgr,
        noisy_tensor,
    ):
        grey_result  = analyse(solid_grey_bgr, solid_grey_tensor)
        noisy_result = analyse(noisy_bgr,      noisy_tensor)
        assert noisy_result.noise_sigma > grey_result.noise_sigma, (
            "Random-noise image should have higher noise_sigma than solid grey"
        )


# ===========================================================================
# TestHistogram
# ===========================================================================

class TestHistogram:

    def test_all_white_histogram_mean_near_255(
        self, all_white_bgr, all_white_tensor
    ):
        result = analyse(all_white_bgr, all_white_tensor)
        assert result.histogram_mean > 240.0, (
            f"All-white histogram_mean should be near 255, "
            f"got {result.histogram_mean:.1f}"
        )

    def test_all_black_histogram_mean_near_0(
        self, all_black_bgr, all_black_tensor
    ):
        result = analyse(all_black_bgr, all_black_tensor)
        assert result.histogram_mean < 15.0, (
            f"All-black histogram_mean should be near 0, "
            f"got {result.histogram_mean:.1f}"
        )

    def test_histogram_std_non_negative(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.histogram_std >= 0.0


# ===========================================================================
# TestDynamicRange
# ===========================================================================

class TestDynamicRange:

    def test_solid_grey_dynamic_range_near_zero(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.dynamic_range_stops == pytest.approx(0.0, abs=0.1), (
            f"Solid-grey image should have zero dynamic range, "
            f"got {result.dynamic_range_stops:.4f} stops"
        )

    def test_dynamic_range_is_non_negative(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.dynamic_range_stops >= 0.0


# ===========================================================================
# TestContrastRms
# ===========================================================================

class TestContrastRms:

    def test_solid_grey_contrast_rms_near_zero(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.contrast_rms == pytest.approx(0.0, abs=1.0), (
            f"Solid-grey image should have zero RMS contrast, "
            f"got {result.contrast_rms:.4f}"
        )

    def test_contrast_rms_is_non_negative(
        self, solid_grey_bgr, solid_grey_tensor
    ):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.contrast_rms >= 0.0


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:

    def test_all_black_no_zero_division_error(
        self, all_black_bgr, all_black_tensor
    ):
        """
        Pure-black images have p1 == 0 in the dynamic-range calculation.
        The implementation must guard against log2(p99 / 0).
        """
        # Must not raise ZeroDivisionError (or any other exception)
        result = analyse(all_black_bgr, all_black_tensor)
        assert isinstance(result, TechnicalScores)

    def test_single_pixel_image_does_not_crash(self):
        """A 1×1 image exercises all quadrant-splitting and stats code paths."""
        bgr    = np.array([[[128, 128, 128]]], dtype=np.uint8)  # (1, 1, 3)
        tensor = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
        result = analyse(bgr, tensor)
        assert isinstance(result, TechnicalScores)

    def test_single_pixel_sharpness_regional_keys(self):
        bgr    = np.array([[[64, 128, 200]]], dtype=np.uint8)
        tensor = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
        result = analyse(bgr, tensor)
        assert set(result.sharpness_regional.keys()) == {
            "top_left", "top_right", "bottom_left", "bottom_right"
        }


# ===========================================================================
# Parametrised smoke test — all fixture images
# ===========================================================================

class TestMusiq:

    def test_musiq_returns_float_or_none(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.musiq is None or isinstance(result.musiq, float), (
            f"musiq must be float or None, got {type(result.musiq)}"
        )

    def test_musiq_in_smoke_analyse(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert isinstance(result, TechnicalScores)


# ===========================================================================
# TestNiqe
# ===========================================================================

class TestNiqe:

    def test_niqe_returns_float_or_none(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert result.niqe is None or isinstance(result.niqe, float), (
            f"niqe must be float or None, got {type(result.niqe)}"
        )

    def test_niqe_stubbed_value_matches_fixture(self, solid_grey_bgr, solid_grey_tensor):
        """The conftest stub returns 4.5 for niqe — verify it flows through."""
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        if result.niqe is not None:
            assert result.niqe == pytest.approx(4.5, abs=0.01)

    def test_niqe_present_in_technical_scores_model(self, solid_grey_bgr, solid_grey_tensor):
        result = analyse(solid_grey_bgr, solid_grey_tensor)
        assert hasattr(result, "niqe")


# ===========================================================================
# TestParallelL1
# ===========================================================================


class TestParallelL1:
    """Parallel ThreadPoolExecutor execution of ML scorers."""

    def test_parallel_l1_all_metrics_present(self, solid_grey_bgr, solid_grey_tensor):
        """All five stubbed metrics should appear in output after parallel execution."""
        scores = analyse(solid_grey_bgr, solid_grey_tensor)
        assert scores.brisque is not None
        assert scores.nima_aesthetic is not None
        assert scores.clip_iqa is not None
        assert scores.musiq is not None
        assert scores.niqe is not None


# ===========================================================================
# TestIsPhotograph
# ===========================================================================


class TestIsPhotograph:
    """_is_photograph() pre-screening gate."""

    def test_blank_image_rejected(self, all_white_bgr, all_white_tensor, monkeypatch):
        import src.analysis.technical as tech_mod

        # Ensure CLIP is skipped (Tier 1 is sufficient for this case)
        monkeypatch.setattr(tech_mod, "_clip_prescreener", False, raising=False)
        monkeypatch.setattr(tech_mod, "_PRESCREENING_ENABLED", True)
        is_photo, reason = _is_photograph(all_white_bgr, all_white_tensor)
        assert is_photo is False
        assert "blank" in reason.lower() or "solid" in reason.lower()

    def test_real_image_passes(self, sharp_edges_bgr, sharp_edges_tensor, monkeypatch):
        """Checkerboard has high std — passes Tier 1; CLIP mocked as unavailable."""
        import src.analysis.technical as tech_mod

        monkeypatch.setattr(tech_mod, "_clip_prescreener", False, raising=False)
        monkeypatch.setattr(tech_mod, "_PRESCREENING_ENABLED", True)
        is_photo, reason = _is_photograph(sharp_edges_bgr, sharp_edges_tensor)
        assert is_photo is True

    def test_prescreening_disabled_always_passes(self, all_white_bgr, all_white_tensor, monkeypatch):
        """When _PRESCREENING_ENABLED is False, all images pass regardless."""
        import src.analysis.technical as tech_mod

        monkeypatch.setattr(tech_mod, "_PRESCREENING_ENABLED", False)
        is_photo, _ = _is_photograph(all_white_bgr, all_white_tensor)
        assert is_photo is True


@pytest.mark.parametrize(
    "image_arrays",
    [
        "solid_grey_image",
        "all_white_image",
        "all_black_image",
        "noisy_image",
        "sharp_edges_image",
    ],
    indirect=True,
    ids=["grey", "white", "black", "noisy", "sharp"],
)
def test_smoke_analyse_all_fixtures(image_arrays):
    """
    End-to-end smoke: *analyse()* must return a valid TechnicalScores for
    every synthetic fixture image without raising any exception.
    """
    bgr, tensor = image_arrays
    result = analyse(bgr, tensor)
    assert isinstance(result, TechnicalScores)
    # Verify the four required structural invariants in one sweep
    assert isinstance(result.brisque, float)
    assert isinstance(result.sharpness_laplacian, float)
    assert isinstance(result.noise_sigma, float)
    assert set(result.sharpness_regional.keys()) == {
        "top_left", "top_right", "bottom_left", "bottom_right"
    }
