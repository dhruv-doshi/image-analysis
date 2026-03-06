from __future__ import annotations

import json
import logging
import math
import re
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

_REPORT_KEYS = (
    "summary", "composition", "aesthetics", "technical",
    "improvements", "editing", "inspiration",
)


def _sanitise_llm_json(raw: str) -> str:
    """Fix common LLM JSON mistakes before parsing.

    1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    2. Leading-plus numbers (+15 → 15) — not valid JSON
    3. Missing comma between fields (]\\n  "key" → ],\\n  "key")
    4. Trailing commas before } or ]
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]           # drop opening fence line
        raw = raw.rsplit("```", 1)[0].strip()  # drop closing fence
    # Extract the JSON object — handles any preamble/epilogue text the LLM emits
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]
    # JSON spec forbids +N numbers
    raw = re.sub(r":\s*\+(\d)", r": \1", raw)
    # Missing comma after ] or } before the next "key": pattern
    raw = re.sub(r'([}\]])\s*\n(\s*"[a-z_]+")', r'\1,\n\2', raw)
    # Trailing commas before closing brace/bracket
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw


def _coerce_report_fields(data: dict) -> dict:
    """Coerce list or dict values in report string fields to plain strings."""
    for key in _REPORT_KEYS:
        val = data.get(key)
        if isinstance(val, list):
            data[key] = "\n".join(str(item) for item in val)
        elif isinstance(val, dict):
            data[key] = "\n".join(f"{k}: {v}" for k, v in val.items())
    return data

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_SYSTEM_PROMPT: str = (_PROMPTS_DIR / "system.md").read_text(encoding="utf-8")

_FEATURE_NAMES: dict[AnalysisFeature, str] = {
    AnalysisFeature.COMPOSITION: "composition",
    AnalysisFeature.AESTHETICS: "aesthetics",
    AnalysisFeature.TECHNICAL: "technical",
    AnalysisFeature.IMPROVEMENTS: "improvements",
    AnalysisFeature.EDITING: "editing",
    AnalysisFeature.INSPIRATION: "inspiration",
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


def _nima_ord(v: float | None) -> int | None:
    if v is None or math.isnan(v):
        return None
    if v >= 7.0:
        return 0
    if v >= 6.0:
        return 1
    if v >= 5.0:
        return 2
    if v >= 4.0:
        return 3
    return 4


def _clip_ord(v: float | None) -> int | None:
    if v is None or math.isnan(v):
        return None
    if v >= 0.60:
        return 0
    if v >= 0.50:
        return 1
    if v >= 0.40:
        return 2
    if v >= 0.30:
        return 3
    return 4


def _musiq_ord(v: float | None) -> int | None:
    if v is None or math.isnan(v):
        return None
    if v >= 70:
        return 0
    if v >= 55:
        return 1
    if v >= 40:
        return 2
    if v >= 25:
        return 3
    return 4


def _niqe_ord(v: float | None) -> int | None:
    if v is None or math.isnan(v):
        return None
    if v < 3:
        return 0
    if v < 5:
        return 1
    if v < 8:
        return 2
    if v < 12:
        return 3
    return 4


def _composition_ord(comp: CompositionScores) -> int:
    """0=excellent … 4=terrible based on balance + subject placement."""
    balance = comp.visual_weight_balance  # 1.0=balanced, >3=concentrated
    rot = comp.rot_alignment_score       # 0=perfect, >0.5=off power points
    penalty = 0
    if balance > 6.0:
        penalty += 2
    elif balance > 4.0:
        penalty += 1
    if rot > 0.6:
        penalty += 1
    return min(4, penalty)


def _compute_quality_tier(tech: TechnicalScores, comp: CompositionScores) -> dict:
    b = _brisque_ord(tech.brisque)
    s = _sharpness_ord(tech.sharpness_laplacian, tech.noise_sigma)
    n = _noise_ord(tech.noise_sigma)
    e = _exposure_ord(
        tech.exposure_clipped_highlights_pct,
        tech.exposure_clipped_shadows_pct,
        tech.histogram_mean,
    )
    na = _nima_ord(tech.nima_aesthetic)
    ci = _clip_ord(tech.clip_iqa)
    mq = _musiq_ord(tech.musiq)
    nq = _niqe_ord(tech.niqe)
    co = _composition_ord(comp)

    # Base weights: BRISQUE 3x, sharpness 2x, noise 2x, exposure 1x = 8
    weighted, weight_sum = b * 3 + s * 2 + n * 2 + e, 8
    if na is not None:
        weighted += na * 2
        weight_sum += 2
    if ci is not None:
        weighted += ci * 1
        weight_sum += 1
    if mq is not None:
        weighted += mq * 1.5
        weight_sum += 1.5
    if nq is not None:
        weighted += nq * 1
        weight_sum += 1
    # Composition: weight 2x
    weighted += co * 2
    weight_sum += 2
    overall = int(weighted / weight_sum + 0.5)

    result = {
        "overall": _t(overall),
        "brisque_tier": _t(b),
        "sharpness_tier": _t(s),
        "noise_tier": _t(n),
        "exposure_tier": _t(e),
        "composition_tier": _t(co),
    }
    if na is not None:
        result["nima_tier"] = _t(na)
    if ci is not None:
        result["clip_tier"] = _t(ci)
    if mq is not None:
        result["musiq_tier"] = _t(mq)
    if nq is not None:
        result["niqe_tier"] = _t(nq)
    return result


def _build_payload(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
    features: AnalysisFeature,
) -> str:
    """Build the annotated JSON user message from scores."""
    requested = [name for flag, name in _FEATURE_NAMES.items() if flag in features]

    payload = {
        "exif": exif.model_dump(exclude_none=True),
        "quality_tier": _compute_quality_tier(tech, comp),
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
            "musiq": {
                "value": tech.musiq,
                "scale": "0–100, higher is better (contrast: BRISQUE is lower-is-better)",
            },
            "niqe": {
                "value": tech.niqe,
                "scale": "lower is better; <3=excellent, 3–5=good, 5–8=average, >8=poor (detects compression/distortion artefacts)",
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
            "horizon_tilt_degrees": {
                "value": comp.horizon_tilt_degrees,
                "scale": "degrees from level; 0=perfectly level, positive=clockwise tilt, null=no clear horizon detected",
            },
            "scene_type": {
                "value": comp.scene_type,
                "scale": "portrait|landscape|architecture|macro|general",
            },
            "color_harmony_type": {
                "value": comp.color_harmony_type,
                "scale": "monochromatic|analogous|complementary|triadic|split-complementary|complex",
            },
            "color_harmony_score": {
                "value": comp.color_harmony_score,
                "scale": "0–1, higher = stronger match to the named template",
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
        openai.APIError: on network or quota failures (caller handles).
        json.JSONDecodeError: if the LLM returns malformed JSON (rare; log + raise).
    """
    client = get_client()
    user_message = _build_payload(tech, comp, exif, features)

    logger.info(
        "LLM request  model=%s  system_prompt=%d chars  user_payload=%d chars  max_tokens=4096",
        _MODEL,
        len(_SYSTEM_PROMPT),
        len(user_message),
    )
    logger.debug("LLM user payload:\n%s", user_message)

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw_text: str = (response.choices[0].message.content or "").strip()
    finish_reason = response.choices[0].finish_reason
    usage = response.usage
    if usage:
        logger.info(
            "LLM response  finish_reason=%s  tokens: prompt=%d  completion=%d  total=%d",
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    else:
        logger.info("LLM response  finish_reason=%s  response=%d chars", finish_reason, len(raw_text))
    logger.debug("LLM raw response:\n%s", raw_text)

    raw_text = _sanitise_llm_json(raw_text)

    if not raw_text:
        raise ValueError(
            f"LLM returned an empty response (finish_reason={response.choices[0].finish_reason})"
        )

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON: %.400s", raw_text)
        raise

    return AnalysisReport(**_coerce_report_fields(data))
