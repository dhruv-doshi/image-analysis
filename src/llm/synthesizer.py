from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from src.llm.client import _MODEL, get_client
from src.models import (
    AnalysisFeature,
    AnalysisReport,
    CompositionScores,
    ExifData,
    TechnicalScores,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_SYSTEM_PROMPT: str = (_PROMPTS_DIR / "system.md").read_text(encoding="utf-8")

_FEATURE_NAMES: dict[AnalysisFeature, str] = {
    AnalysisFeature.COMPOSITION:  "composition",
    AnalysisFeature.AESTHETICS:   "aesthetics",
    AnalysisFeature.TECHNICAL:    "technical",
    AnalysisFeature.IMPROVEMENTS: "improvements",
    AnalysisFeature.EDITING:      "editing",
    AnalysisFeature.INSPIRATION:  "inspiration",
}


_TIER_LABELS = ["excellent", "good", "average", "poor", "terrible"]


def _t(ordinal: int) -> str:
    return _TIER_LABELS[max(0, min(4, ordinal))]


def _brisque_ord(v: float) -> int:
    if math.isnan(v):
        return 2
    if v < 30:
        return 0
    if v < 50:
        return 1
    if v < 65:
        return 2
    if v < 80:
        return 3
    return 4


def _sharpness_ord(sharpness: float, noise_sigma: float) -> int:
    adj = max(0.0, sharpness - noise_sigma**2 * 20)
    if adj > 500:
        return 0
    if adj >= 200:
        return 2
    if adj >= 80:
        return 3
    return 4


def _noise_ord(v: float) -> int:
    if math.isnan(v):
        return 2
    if v < 3:
        return 0
    if v <= 8:
        return 2
    if v <= 15:
        return 3
    return 4


def _exposure_ord(hl: float, sh: float, mean: float) -> int:
    severe = sum([hl > 15, sh > 20, mean < 40 or mean > 230])
    poor = sum([hl > 5, sh > 5, mean < 80 or mean > 200])
    if severe >= 2:
        return 4
    if severe >= 1:
        return 3
    if poor >= 2:
        return 3
    if poor >= 1:
        return 2
    if hl > 2 or sh > 2:
        return 1
    return 0


def _compute_quality_tier(tech: TechnicalScores) -> dict:
    b = _brisque_ord(tech.brisque)
    s = _sharpness_ord(tech.sharpness_laplacian, tech.noise_sigma)
    n = _noise_ord(tech.noise_sigma)
    e = _exposure_ord(
        tech.exposure_clipped_highlights_pct,
        tech.exposure_clipped_shadows_pct,
        tech.histogram_mean,
    )
    overall = int((b * 3 + s * 2 + n * 2 + e * 1) / 8.0 + 0.5)
    return {
        "overall":        _t(overall),
        "brisque_tier":   _t(b),
        "sharpness_tier": _t(s),
        "noise_tier":     _t(n),
        "exposure_tier":  _t(e),
    }


def _build_payload(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
    features: AnalysisFeature,
) -> str:
    """Build the annotated JSON user message from scores."""
    requested = [
        name for flag, name in _FEATURE_NAMES.items() if flag in features
    ]

    payload = {
        "exif": exif.model_dump(exclude_none=True),
        "quality_tier": _compute_quality_tier(tech),

        "technical": {
            "brisque": {
                "value": tech.brisque,
                "scale": "0–100, lower is better; <30=excellent, 30–50=good, >70=poor",
            },
            "nima_aesthetic": {
                "value": tech.nima_aesthetic,
                "scale": "1–10, higher is better; <5=poor, 5–7=average, >7=good",
            },
            "clip_iqa": {
                "value": tech.clip_iqa,
                "scale": "0–1, higher is better; >0.6=good perceptual quality",
            },
            "sharpness_laplacian": {
                "value": tech.sharpness_laplacian,
                "scale": "Laplacian variance; <200=blurry, 200–500=acceptable, >500=sharp",
            },
            "sharpness_regional": {
                "value": tech.sharpness_regional,
                "scale": "per-quadrant Laplacian variance; same scale as global sharpness",
            },
            "noise_sigma": {
                "value": tech.noise_sigma,
                "scale": "estimated Gaussian noise std dev; <3=clean, 3–8=moderate, >8=noisy",
            },
            "exposure_highlights_pct": {
                "value": tech.exposure_clipped_highlights_pct,
                "scale": "% pixels clipped at 255; <2=acceptable, 2–5=caution, >5=overexposed",
            },
            "exposure_shadows_pct": {
                "value": tech.exposure_clipped_shadows_pct,
                "scale": "% pixels clipped at 0; <2=acceptable, >5=underexposed",
            },
            "histogram_mean": {
                "value": tech.histogram_mean,
                "scale": "mean luminance 0–255; <80=dark, 80–170=well-exposed, >200=very bright",
            },
            "histogram_std": {
                "value": tech.histogram_std,
                "scale": "luminance spread; <30=flat/low contrast, >80=wide tonal range",
            },
            "dynamic_range_stops": {
                "value": tech.dynamic_range_stops,
                "scale": "log2(p99/p1); <3=flat/low dynamic range, 3–5=moderate, >5=rich",
            },
            "contrast_rms": {
                "value": tech.contrast_rms,
                "scale": "std/mean of grayscale; <0.2=flat, 0.2–0.5=normal, >0.5=punchy",
            },
        },

        "composition": {
            "saliency_centroid_x": {
                "value": comp.saliency_centroid_x,
                "scale": "0=left edge, 1=right edge; 0.333 and 0.667 are rule-of-thirds lines",
            },
            "saliency_centroid_y": {
                "value": comp.saliency_centroid_y,
                "scale": "0=top edge, 1=bottom edge; 0.333 and 0.667 are rule-of-thirds lines",
            },
            "rot_alignment_score": {
                "value": comp.rot_alignment_score,
                "scale": "0=perfectly on a RoT power point, 1=worst; <0.15=excellent",
            },
            "golden_ratio_alignment_score": {
                "value": comp.golden_ratio_alignment_score,
                "scale": "0=perfectly on a Golden Ratio power point, 1=worst; <0.15=excellent",
            },
            "best_alignment": {
                "value": comp.best_alignment,
                "scale": "rule_of_thirds | golden_ratio — which framework the subject aligns to",
            },
            "negative_space_ratio": {
                "value": comp.negative_space_ratio,
                "scale": "0–1 fraction of non-salient pixels; >0.5=minimalist composition",
            },
            "visual_weight_quadrants": {
                "value": comp.visual_weight_quadrants,
                "scale": "per-quadrant saliency fraction; sums to 1.0",
            },
            "visual_weight_balance": {
                "value": comp.visual_weight_balance,
                "scale": "heaviest/lightest quadrant ratio; 1.0=balanced, >3=heavily concentrated",
            },
            "symmetry_horizontal": {
                "value": comp.symmetry_horizontal,
                "scale": "0–1 NCC of left vs right halves; 1.0=perfectly left-right symmetric",
            },
            "symmetry_vertical": {
                "value": comp.symmetry_vertical,
                "scale": "0–1 NCC of top vs bottom halves; 1.0=perfectly top-bottom symmetric",
            },
            "leading_lines_converge_to_subject": {
                "value": comp.leading_lines_converge_to_subject,
                "scale": "bool; True=detected lines guide the eye toward the subject",
            },
            "line_pattern": {
                "value": comp.line_pattern,
                "scale": "diagonal=dynamic, horizontal=calm/stable, vertical=formal, mixed, none",
            },
            "dominant_line_angles": {
                "value": comp.dominant_line_angles,
                "scale": "line angles in degrees 0–180; 0/180=horizontal, 90=vertical",
            },
        },

        "requested_features": requested,
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


def synthesise(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
    features: AnalysisFeature = AnalysisFeature.FULL,
) -> AnalysisReport:
    """
    Synthesise a natural-language photo critique via Claude.

    Args:
        tech:     Layer 1 technical scores.
        comp:     Layer 2 composition scores.
        exif:     EXIF metadata (may have None fields).
        features: Which AnalysisFeature flags to include in the report.

    Returns:
        AnalysisReport Pydantic instance with populated fields for each
        requested feature plus always a summary.

    Raises:
        anthropic.APIError: on network or quota failures (caller handles).
        json.JSONDecodeError: if Claude returns malformed JSON (rare; log + raise).
    """
    client = get_client()
    user_message = _build_payload(tech, comp, exif, features)

    logger.debug("Calling Claude %s with %d-byte payload", _MODEL, len(user_message))

    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text: str = response.content[0].text.strip()
    logger.debug(
        "Claude responded with %d chars (stop_reason=%s)",
        len(raw_text),
        response.stop_reason,
    )

    # Strip markdown code fences if Claude wrapped the JSON
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]  # drop opening fence line
        raw_text = raw_text.rsplit("```", 1)[0].strip()  # drop closing fence

    if not raw_text:
        raise ValueError(
            f"Claude returned an empty response (stop_reason={response.stop_reason})"
        )

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Claude returned non-JSON: %.400s", raw_text)
        raise

    # Claude occasionally returns list values for string fields; join them.
    for key in ("summary", "composition", "aesthetics", "technical",
                "improvements", "editing", "inspiration"):
        if isinstance(data.get(key), list):
            data[key] = "\n".join(str(item) for item in data[key])

    return AnalysisReport(**data)
