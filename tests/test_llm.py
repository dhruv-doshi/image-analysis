from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# Mock openai at sys.modules level BEFORE any src.llm imports
_openai_mock = MagicMock()
sys.modules.setdefault("openai", _openai_mock)

from src.models import (
    AnalysisFeature,
    AnalysisReport,
    CompositionScores,
    ExifData,
    TechnicalScores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(json_body: dict):
    """Construct a mock chat.completions.create() return value."""
    choice = MagicMock()
    choice.message.content = json.dumps(json_body)
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_FULL_RESPONSE = {
    "summary": "A crisp, well-exposed landscape with strong RoT placement.",
    "composition": "The subject sits near the top-left RoT intersection.",
    "aesthetics": "Warm tones and high contrast give a golden-hour feel.",
    "technical": "Sharpness is excellent; minor highlight clipping.",
    "improvements": "1. Try a lower angle. 2. Use graduated ND filter.",
    "editing": "Lightroom: Highlights -30, Clarity +10.",
    "inspiration": "Study Ansel Adams' tonal range work.",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_tech() -> TechnicalScores:
    """Minimal valid TechnicalScores with no optional fields."""
    return TechnicalScores(
        brisque=35.0,
        sharpness_laplacian=520.0,
        sharpness_regional={
            "top_left": 500.0, "top_right": 540.0,
            "bottom_left": 510.0, "bottom_right": 530.0,
        },
        noise_sigma=2.1,
        exposure_clipped_highlights_pct=1.2,
        exposure_clipped_shadows_pct=0.4,
        histogram_mean=142.0,
        histogram_std=58.0,
        dynamic_range_stops=4.8,
        contrast_rms=0.38,
        niqe=4.5,
    )


@pytest.fixture()
def minimal_comp() -> CompositionScores:
    """Minimal valid CompositionScores."""
    return CompositionScores(
        saliency_centroid_x=0.35,
        saliency_centroid_y=0.33,
        rot_alignment_score=0.06,
        golden_ratio_alignment_score=0.21,
        best_alignment="rule_of_thirds",
        negative_space_ratio=0.60,
        visual_weight_quadrants={
            "top_left": 0.42, "top_right": 0.18,
            "bottom_left": 0.25, "bottom_right": 0.15,
        },
        visual_weight_balance=2.8,
        symmetry_horizontal=0.42,
        symmetry_vertical=0.38,
        dominant_line_angles=[15.0, 165.0],
        leading_lines_converge_to_subject=True,
        line_pattern="diagonal",
    )


@pytest.fixture()
def empty_exif() -> ExifData:
    return ExifData()


@pytest.fixture()
def full_exif() -> ExifData:
    return ExifData(
        camera_make="Canon",
        camera_model="EOS R5",
        iso=400,
        shutter_speed="1/250",
        aperture=4.0,
        focal_length=85.0,
        lens_model="RF 85mm f/1.2",
        image_width=4000,
        image_height=6000,
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_comp(balance: float, rot: float) -> CompositionScores:
    """Build a minimal CompositionScores with specific balance and RoT values."""
    return CompositionScores(
        saliency_centroid_x=0.33,
        saliency_centroid_y=0.33,
        rot_alignment_score=rot,
        golden_ratio_alignment_score=rot + 0.1,
        best_alignment="rule_of_thirds",
        negative_space_ratio=0.5,
        visual_weight_quadrants={
            "top_left": 0.25, "top_right": 0.25,
            "bottom_left": 0.25, "bottom_right": 0.25,
        },
        visual_weight_balance=balance,
        symmetry_horizontal=0.5,
        symmetry_vertical=0.5,
        dominant_line_angles=[],
        leading_lines_converge_to_subject=False,
        line_pattern="none",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputType:
    """synthesise() must return an AnalysisReport instance."""

    def test_returns_analysis_report(self, minimal_tech, minimal_comp, empty_exif):
        from unittest.mock import MagicMock, patch
        import src.llm.synthesizer as syn

        with patch("src.llm.synthesizer.get_client") as mock_get:
            mc = MagicMock()
            mock_get.return_value = mc
            mc.chat.completions.create.return_value = _fake_response(_FULL_RESPONSE)
            result = syn.synthesise(minimal_tech, minimal_comp, empty_exif)

        assert isinstance(result, AnalysisReport)

    def test_summary_is_always_present(self, minimal_tech, minimal_comp, empty_exif):
        from unittest.mock import MagicMock, patch
        import src.llm.synthesizer as syn

        with patch("src.llm.synthesizer.get_client") as mock_get:
            mc = MagicMock()
            mock_get.return_value = mc
            mc.chat.completions.create.return_value = _fake_response({"summary": "Good shot."})
            result = syn.synthesise(
                minimal_tech, minimal_comp, empty_exif,
                features=AnalysisFeature.COMPOSITION,
            )

        assert result.summary == "Good shot."


class TestFeatureFlags:
    """Feature flags control which sections are populated."""

    @pytest.mark.parametrize("flag,field", [
        (AnalysisFeature.COMPOSITION,  "composition"),
        (AnalysisFeature.AESTHETICS,   "aesthetics"),
        (AnalysisFeature.TECHNICAL,    "technical"),
        (AnalysisFeature.IMPROVEMENTS, "improvements"),
        (AnalysisFeature.EDITING,      "editing"),
        (AnalysisFeature.INSPIRATION,  "inspiration"),
    ])
    def test_single_feature_populates_only_that_field(
        self, flag, field, minimal_tech, minimal_comp, empty_exif
    ):
        from unittest.mock import MagicMock, patch
        import src.llm.synthesizer as syn

        response_body = {"summary": "OK.", field: "Some critique."}

        with patch("src.llm.synthesizer.get_client") as mock_get:
            mc = MagicMock()
            mock_get.return_value = mc
            mc.chat.completions.create.return_value = _fake_response(response_body)
            result = syn.synthesise(
                minimal_tech, minimal_comp, empty_exif, features=flag
            )

        assert getattr(result, field) == "Some critique."
        for other in ["composition", "aesthetics", "technical",
                      "improvements", "editing", "inspiration"]:
            if other != field:
                assert getattr(result, other) is None, (
                    f"Expected {other}=None when only {field} requested"
                )


class TestPayloadContent:
    """The JSON sent to the LLM contains all expected score fields."""

    def _capture_payload(self, tech, comp, exif, features=AnalysisFeature.FULL):
        from unittest.mock import MagicMock, patch
        import src.llm.synthesizer as syn

        with patch("src.llm.synthesizer.get_client") as mock_get:
            mc = MagicMock()
            mock_get.return_value = mc
            mc.chat.completions.create.return_value = _fake_response({"summary": "ok"})
            syn.synthesise(tech, comp, exif, features)
            call_kwargs = mc.chat.completions.create.call_args

        # messages[0] is the system prompt; user payload is at index 1
        user_content = call_kwargs.kwargs["messages"][1]["content"]
        return json.loads(user_content)

    def test_payload_contains_technical_scores(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert "technical" in payload
        assert "brisque" in payload["technical"]
        assert payload["technical"]["brisque"]["value"] == pytest.approx(35.0)

    def test_payload_contains_composition_scores(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert "composition" in payload
        assert "rot_alignment_score" in payload["composition"]

    def test_payload_contains_requested_features_list(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(
            minimal_tech, minimal_comp, empty_exif,
            features=AnalysisFeature.COMPOSITION | AnalysisFeature.TECHNICAL,
        )
        assert set(payload["requested_features"]) == {"composition", "technical"}

    def test_exif_none_fields_excluded_from_payload(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert payload["exif"] == {}

    def test_exif_fields_present_when_populated(self, minimal_tech, minimal_comp, full_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, full_exif)
        assert payload["exif"]["camera_make"] == "Canon"
        assert payload["exif"]["iso"] == 400

    def test_all_scales_are_strings(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        for section in ("technical", "composition"):
            for key, entry in payload[section].items():
                if isinstance(entry, dict) and "scale" in entry:
                    assert isinstance(entry["scale"], str), (
                        f"{section}.{key}.scale must be a string"
                    )

    def test_niqe_key_present_in_payload(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert "niqe" in payload["technical"]

    def test_niqe_scale_is_string(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert isinstance(payload["technical"]["niqe"]["scale"], str)

    def test_quality_tier_has_composition_tier(self, minimal_tech, minimal_comp, empty_exif):
        payload = self._capture_payload(minimal_tech, minimal_comp, empty_exif)
        assert "composition_tier" in payload["quality_tier"]


class TestErrorHandling:
    """Malformed JSON from the LLM is re-raised as JSONDecodeError."""

    def test_invalid_json_raises(self, minimal_tech, minimal_comp, empty_exif):
        from unittest.mock import MagicMock, patch
        import src.llm.synthesizer as syn

        bad_choice = MagicMock()
        bad_choice.message.content = "This is not JSON {{"
        bad_choice.finish_reason = "stop"
        bad_resp = MagicMock()
        bad_resp.choices = [bad_choice]

        with patch("src.llm.synthesizer.get_client") as mock_get:
            mc = MagicMock()
            mock_get.return_value = mc
            mc.chat.completions.create.return_value = bad_resp
            with pytest.raises(json.JSONDecodeError):
                syn.synthesise(minimal_tech, minimal_comp, empty_exif)


class TestMusiqOrd:
    """_musiq_ord() maps MUSIQ scores (0–100, higher=better) to 0–4 ordinals."""

    @pytest.mark.parametrize("val,expected", [
        (80.0, 0),  # >= 70 → excellent
        (60.0, 1),  # >= 55 → good
        (45.0, 2),  # >= 40 → average
        (30.0, 3),  # >= 25 → poor
        (10.0, 4),  # < 25  → terrible
    ])
    def test_ordinal_boundaries(self, val, expected):
        from src.llm.synthesizer import _musiq_ord
        assert _musiq_ord(val) == expected

    def test_none_returns_none(self):
        from src.llm.synthesizer import _musiq_ord
        assert _musiq_ord(None) is None

    def test_nan_returns_none(self):
        from src.llm.synthesizer import _musiq_ord
        assert _musiq_ord(float("nan")) is None


class TestNiqeOrd:
    """_niqe_ord() maps NIQE scores (lower=better) to 0–4 ordinals."""

    @pytest.mark.parametrize("val,expected", [
        (2.0,  0),  # < 3   → excellent
        (4.0,  1),  # < 5   → good
        (6.0,  2),  # < 8   → average
        (10.0, 3),  # < 12  → poor
        (15.0, 4),  # >= 12 → terrible
    ])
    def test_ordinal_boundaries(self, val, expected):
        from src.llm.synthesizer import _niqe_ord
        assert _niqe_ord(val) == expected

    def test_none_returns_none(self):
        from src.llm.synthesizer import _niqe_ord
        assert _niqe_ord(None) is None

    def test_nan_returns_none(self):
        from src.llm.synthesizer import _niqe_ord
        assert _niqe_ord(float("nan")) is None


class TestCompositionOrd:
    """_composition_ord() blends visual_weight_balance + rot_alignment_score into 0–4."""

    def test_balanced_good_rot_returns_zero(self):
        from src.llm.synthesizer import _composition_ord
        assert _composition_ord(_make_comp(balance=1.5, rot=0.1)) == 0

    def test_moderate_imbalance_adds_one_penalty(self):
        from src.llm.synthesizer import _composition_ord
        # balance > 4 → +1, rot fine → total 1
        assert _composition_ord(_make_comp(balance=5.0, rot=0.1)) == 1

    def test_heavy_imbalance_adds_two_penalties(self):
        from src.llm.synthesizer import _composition_ord
        # balance > 6 → +2, rot fine → total 2
        assert _composition_ord(_make_comp(balance=7.0, rot=0.1)) == 2

    def test_bad_rot_adds_one_penalty(self):
        from src.llm.synthesizer import _composition_ord
        # balance fine, rot > 0.6 → +1
        assert _composition_ord(_make_comp(balance=1.5, rot=0.8)) == 1

    def test_heavy_imbalance_and_bad_rot_sums_penalties(self):
        from src.llm.synthesizer import _composition_ord
        # balance > 6 → +2, rot > 0.6 → +1 → total 3
        assert _composition_ord(_make_comp(balance=8.0, rot=0.9)) == 3

    def test_result_never_exceeds_four(self):
        from src.llm.synthesizer import _composition_ord
        # Worst possible inputs — penalty capped at 4
        assert _composition_ord(_make_comp(balance=99.0, rot=0.99)) <= 4


class TestQualityTierNewMetrics:
    """_compute_quality_tier() integration tests for MUSIQ, NIQE, composition."""

    def test_composition_tier_always_present(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        result = _compute_quality_tier(minimal_tech, minimal_comp)
        assert "composition_tier" in result

    def test_composition_tier_is_valid_label(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        result = _compute_quality_tier(minimal_tech, minimal_comp)
        assert result["composition_tier"] in ("excellent", "good", "average", "poor", "terrible")

    def test_bad_composition_reflected_in_composition_tier(self, minimal_tech):
        from src.llm.synthesizer import _compute_quality_tier
        bad_comp = _make_comp(balance=8.0, rot=0.9)
        result = _compute_quality_tier(minimal_tech, bad_comp)
        # penalty = 3 → "poor"
        assert result["composition_tier"] == "poor"

    def test_musiq_tier_included_when_provided(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        tech = minimal_tech.model_copy(update={"musiq": 75.0})
        result = _compute_quality_tier(tech, minimal_comp)
        assert "musiq_tier" in result
        assert result["musiq_tier"] == "excellent"  # 75 >= 70

    def test_musiq_tier_absent_when_none(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        tech = minimal_tech.model_copy(update={"musiq": None})
        result = _compute_quality_tier(tech, minimal_comp)
        assert "musiq_tier" not in result

    def test_niqe_tier_included_when_provided(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        tech = minimal_tech.model_copy(update={"niqe": 4.0})
        result = _compute_quality_tier(tech, minimal_comp)
        assert "niqe_tier" in result
        assert result["niqe_tier"] == "good"  # 3.0 ≤ 4.0 < 5.0

    def test_niqe_tier_absent_when_none(self, minimal_tech, minimal_comp):
        from src.llm.synthesizer import _compute_quality_tier
        tech = minimal_tech.model_copy(update={"niqe": None})
        result = _compute_quality_tier(tech, minimal_comp)
        assert "niqe_tier" not in result

    def test_overall_tier_worsens_with_bad_composition(self, minimal_tech):
        from src.llm.synthesizer import _compute_quality_tier
        good_comp = _make_comp(balance=1.5, rot=0.1)
        bad_comp  = _make_comp(balance=8.0, rot=0.9)
        good_result = _compute_quality_tier(minimal_tech, good_comp)
        bad_result  = _compute_quality_tier(minimal_tech, bad_comp)
        _labels = ["excellent", "good", "average", "poor", "terrible"]
        assert _labels.index(bad_result["overall"]) >= _labels.index(good_result["overall"])


@pytest.mark.parametrize("features", [
    AnalysisFeature.COMPOSITION,
    AnalysisFeature.TECHNICAL,
    AnalysisFeature.IMPROVEMENTS,
    AnalysisFeature.FULL,
    AnalysisFeature.COMPOSITION | AnalysisFeature.EDITING,
], ids=["composition", "technical", "improvements", "full", "comp+editing"])
def test_synthesise_smoke(features, minimal_tech, minimal_comp, empty_exif):
    """synthesise() completes without error for all feature flag combinations."""
    from unittest.mock import MagicMock, patch
    import src.llm.synthesizer as syn

    requested_names = [
        name for flag, name in syn._FEATURE_NAMES.items() if flag in features
    ]
    response_body = {"summary": "Smoke test."} | {n: "ok" for n in requested_names}

    with patch("src.llm.synthesizer.get_client") as mock_get:
        mc = MagicMock()
        mock_get.return_value = mc
        mc.chat.completions.create.return_value = _fake_response(response_body)
        result = syn.synthesise(minimal_tech, minimal_comp, empty_exif, features)

    assert isinstance(result, AnalysisReport)
    assert result.summary == "Smoke test."
