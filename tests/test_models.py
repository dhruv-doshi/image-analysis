"""
Unit tests for Pydantic data models defined in *src/models.py*.

Covers:
  - TechnicalScores  — field validation, optional fields, dict structure
  - ExifData         — all-None construction, type enforcement, serialisation
  - AnalysisFeature  — Flag enum, FULL composed of exactly 6 atomic flags
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import AnalysisFeature, ExifData, TechnicalScores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_scores(**overrides) -> dict:
    """Return a complete set of valid kwargs for TechnicalScores."""
    base = {
        "brisque": 50.0,
        "sharpness_laplacian": 120.0,
        "sharpness_regional": {
            "top_left":     100.0,
            "top_right":    110.0,
            "bottom_left":   90.0,
            "bottom_right": 105.0,
        },
        "noise_sigma": 3.5,
        "exposure_clipped_highlights_pct": 0.5,
        "exposure_clipped_shadows_pct":    0.2,
        "histogram_mean": 130.0,
        "histogram_std":   40.0,
        "dynamic_range_stops": 5.0,
        "contrast_rms": 25.0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# TechnicalScores
# ===========================================================================

class TestTechnicalScores:

    def test_valid_construction(self):
        ts = TechnicalScores(**_valid_scores())
        assert isinstance(ts, TechnicalScores)

    def test_required_fields_stored_correctly(self):
        ts = TechnicalScores(**_valid_scores())
        assert ts.brisque == pytest.approx(50.0)
        assert ts.noise_sigma == pytest.approx(3.5)

    def test_rejects_non_coercible_string_for_brisque(self):
        with pytest.raises(ValidationError):
            TechnicalScores(**_valid_scores(brisque="not_a_number"))

    def test_rejects_non_coercible_string_for_sharpness_laplacian(self):
        with pytest.raises(ValidationError):
            TechnicalScores(**_valid_scores(sharpness_laplacian="sharp"))

    def test_nima_aesthetic_defaults_to_none(self):
        ts = TechnicalScores(**_valid_scores())
        assert ts.nima_aesthetic is None

    def test_clip_iqa_defaults_to_none(self):
        ts = TechnicalScores(**_valid_scores())
        assert ts.clip_iqa is None

    def test_accepts_none_for_nima_aesthetic(self):
        ts = TechnicalScores(**_valid_scores(nima_aesthetic=None))
        assert ts.nima_aesthetic is None

    def test_accepts_none_for_clip_iqa(self):
        ts = TechnicalScores(**_valid_scores(clip_iqa=None))
        assert ts.clip_iqa is None

    def test_sets_nima_aesthetic_when_provided(self):
        ts = TechnicalScores(**_valid_scores(nima_aesthetic=7.2))
        assert ts.nima_aesthetic == pytest.approx(7.2)

    def test_sets_clip_iqa_when_provided(self):
        ts = TechnicalScores(**_valid_scores(clip_iqa=0.85))
        assert ts.clip_iqa == pytest.approx(0.85)

    def test_sharpness_regional_is_dict(self):
        ts = TechnicalScores(**_valid_scores())
        assert isinstance(ts.sharpness_regional, dict)

    def test_sharpness_regional_has_exactly_four_keys(self):
        ts = TechnicalScores(**_valid_scores())
        assert set(ts.sharpness_regional.keys()) == {
            "top_left", "top_right", "bottom_left", "bottom_right"
        }

    def test_model_dump_round_trips_all_required_fields(self):
        data = _valid_scores()
        ts = TechnicalScores(**data)
        dumped = ts.model_dump()
        assert dumped["brisque"] == pytest.approx(50.0)
        assert dumped["dynamic_range_stops"] == pytest.approx(5.0)

    def test_model_dump_exclude_none_omits_unset_optionals(self):
        ts = TechnicalScores(**_valid_scores())
        dumped = ts.model_dump(exclude_none=True)
        assert "nima_aesthetic" not in dumped
        assert "clip_iqa" not in dumped

    def test_model_dump_exclude_none_keeps_set_optionals(self):
        ts = TechnicalScores(**_valid_scores(nima_aesthetic=6.5, clip_iqa=0.75))
        dumped = ts.model_dump(exclude_none=True)
        assert "nima_aesthetic" in dumped
        assert "clip_iqa" in dumped


# ===========================================================================
# ExifData
# ===========================================================================

class TestExifData:

    def test_all_none_construction_succeeds(self):
        exif = ExifData()
        assert isinstance(exif, ExifData)

    def test_all_fields_none_after_empty_construction(self):
        exif = ExifData()
        assert exif.iso is None

    def test_accepts_valid_int_iso(self):
        exif = ExifData(iso=400)
        assert exif.iso == 400

    def test_rejects_non_coercible_iso(self):
        with pytest.raises(ValidationError):
            ExifData(iso="two hundred")

    def test_accepts_image_dimensions(self):
        exif = ExifData(image_width=1920, image_height=1080)
        assert exif.image_width == 1920
        assert exif.image_height == 1080

    def test_model_dump_exclude_none_only_keeps_set_fields(self):
        exif = ExifData(image_width=800, image_height=600)
        dumped = exif.model_dump(exclude_none=True)
        assert "image_width" in dumped
        assert "image_height" in dumped
        assert "iso" not in dumped
        assert "aperture" not in dumped

    def test_model_dump_exclude_none_empty_when_all_none(self):
        exif = ExifData()
        dumped = exif.model_dump(exclude_none=True)
        assert dumped == {}

    def test_model_dump_include_none_lists_all_fields(self):
        exif = ExifData()
        dumped = exif.model_dump()
        # All declared fields should be present, values are None
        assert "iso" in dumped
        assert "image_width" in dumped
        assert "image_height" in dumped


# ===========================================================================
# AnalysisFeature
# ===========================================================================

class TestAnalysisFeature:

    @staticmethod
    def _atomic_flags() -> list:
        """Return all single-bit (atomic) members of AnalysisFeature."""
        return [
            m for m in AnalysisFeature
            if int(m.value) > 0 and bin(int(m.value)).count("1") == 1
        ]

    def test_full_member_exists(self):
        assert hasattr(AnalysisFeature, "FULL")

    def test_full_is_non_zero(self):
        assert int(AnalysisFeature.FULL.value) != 0

    def test_exactly_six_atomic_flags(self):
        atomic = self._atomic_flags()
        assert len(atomic) == 6, (
            f"Expected 6 atomic feature flags, found {len(atomic)}: {atomic}"
        )

    def test_full_includes_every_atomic_flag(self):
        for flag in self._atomic_flags():
            assert flag in AnalysisFeature.FULL, (
                f"{flag!r} is not contained in AnalysisFeature.FULL"
            )

    def test_full_bit_count_equals_six(self):
        assert bin(int(AnalysisFeature.FULL.value)).count("1") == 6
