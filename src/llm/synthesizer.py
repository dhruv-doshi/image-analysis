from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from src.config.metrics import get_weight, is_enabled
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
    "summary",
    "composition",
    "aesthetics",
    "technical",
    "improvements",
    "editing",
    "inspiration",
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
        raw = raw.split("\n", 1)[-1]  # drop opening fence line
        raw = raw.rsplit("```", 1)[0].strip()  # drop closing fence
    # Extract the JSON object — handles any preamble/epilogue text the LLM emits
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    # JSON spec forbids +N numbers
    raw = re.sub(r":\s*\+(\d)", r": \1", raw)
    # Missing comma after ] or } before the next "key": pattern
    raw = re.sub(r'([}\]])\s*\n(\s*"[a-z_]+")', r"\1,\n\2", raw)
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
    rot = comp.rot_alignment_score  # 0=perfect, >0.5=off power points
    penalty = 0
    if balance > 6.0:
        penalty += 2
    elif balance > 4.0:
        penalty += 1
    if rot > 0.6:
        penalty += 1
    return min(4, penalty)


def _compute_quality_tier(tech: TechnicalScores, comp: CompositionScores) -> dict:
    weighted, weight_sum = 0.0, 0.0

    b = None
    if is_enabled("brisque") and tech.brisque is not None:
        b = _brisque_ord(tech.brisque)
        w = get_weight("brisque")
        if w > 0:
            weighted += b * w
            weight_sum += w

    s = None
    if is_enabled("sharpness_laplacian") and tech.sharpness_laplacian is not None:
        noise_adj = tech.noise_sigma if tech.noise_sigma is not None else 0.0
        s = _sharpness_ord(tech.sharpness_laplacian, noise_adj)
        w = get_weight("sharpness_laplacian")
        if w > 0:
            weighted += s * w
            weight_sum += w

    n = None
    if is_enabled("noise_sigma") and tech.noise_sigma is not None:
        n = _noise_ord(tech.noise_sigma)
        w = get_weight("noise_sigma")
        if w > 0:
            weighted += n * w
            weight_sum += w

    e = None
    if (
        is_enabled("exposure_clipped_highlights_pct")
        and tech.exposure_clipped_highlights_pct is not None
        and tech.exposure_clipped_shadows_pct is not None
        and tech.histogram_mean is not None
    ):
        e = _exposure_ord(
            tech.exposure_clipped_highlights_pct,
            tech.exposure_clipped_shadows_pct,
            tech.histogram_mean,
        )
        w = get_weight("exposure_clipped_highlights_pct")
        if w > 0:
            weighted += e * w
            weight_sum += w

    na = _nima_ord(tech.nima_aesthetic) if is_enabled("nima_aesthetic") else None
    if na is not None:
        w = get_weight("nima_aesthetic")
        if w > 0:
            weighted += na * w
            weight_sum += w

    ci = _clip_ord(tech.clip_iqa) if is_enabled("clip_iqa") else None
    if ci is not None:
        w = get_weight("clip_iqa")
        if w > 0:
            weighted += ci * w
            weight_sum += w

    mq = _musiq_ord(tech.musiq) if is_enabled("musiq") else None
    if mq is not None:
        w = get_weight("musiq")
        if w > 0:
            weighted += mq * w
            weight_sum += w

    nq = _niqe_ord(tech.niqe) if is_enabled("niqe") else None
    if nq is not None:
        w = get_weight("niqe")
        if w > 0:
            weighted += nq * w
            weight_sum += w

    co = None
    if (
        is_enabled("rot_alignment_score")
        and comp.rot_alignment_score is not None
        and is_enabled("visual_weight_balance")
        and comp.visual_weight_balance is not None
    ):
        co = _composition_ord(comp)
        w = get_weight("rot_alignment_score")
        if w > 0:
            weighted += co * w
            weight_sum += w

    overall = int(weighted / weight_sum + 0.5) if weight_sum > 0 else 2

    result: dict = {"overall": _t(overall)}
    if b is not None:
        result["brisque_tier"] = _t(b)
    if s is not None:
        result["sharpness_tier"] = _t(s)
    if n is not None:
        result["noise_tier"] = _t(n)
    if e is not None:
        result["exposure_tier"] = _t(e)
    if co is not None:
        result["composition_tier"] = _t(co)
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

    def _add(d: dict, payload_key: str, metric_key: str, value, scale: str) -> None:
        if is_enabled(metric_key) and value is not None:
            d[payload_key] = {"value": value, "scale": scale}

    tech_section: dict = {}
    _add(
        tech_section,
        "brisque",
        "brisque",
        tech.brisque,
        "0–100, lower is better; <30=excellent, 30–50=good, >70=poor",
    )
    _add(
        tech_section,
        "nima_aesthetic",
        "nima_aesthetic",
        tech.nima_aesthetic,
        "1–10, higher is better; <5=poor, 5–7=average, >7=good",
    )
    _add(
        tech_section,
        "clip_iqa",
        "clip_iqa",
        tech.clip_iqa,
        "0–1, higher is better; >0.6=good perceptual quality",
    )
    _add(
        tech_section,
        "musiq",
        "musiq",
        tech.musiq,
        "0–100, higher is better (contrast: BRISQUE is lower-is-better)",
    )
    _add(
        tech_section,
        "niqe",
        "niqe",
        tech.niqe,
        "lower is better; <3=excellent, 3–5=good, 5–8=average, >8=poor (detects compression/distortion artefacts)",
    )
    _add(
        tech_section,
        "sharpness_laplacian",
        "sharpness_laplacian",
        tech.sharpness_laplacian,
        "Laplacian variance; <200=blurry, 200–500=acceptable, >500=sharp",
    )
    if is_enabled("sharpness_regional") and tech.sharpness_regional:
        tech_section["sharpness_regional"] = {
            "value": tech.sharpness_regional,
            "scale": "per-quadrant Laplacian variance; same scale as global sharpness",
        }
    _add(
        tech_section,
        "noise_sigma",
        "noise_sigma",
        tech.noise_sigma,
        "estimated Gaussian noise std dev; <3=clean, 3–8=moderate, >8=noisy",
    )
    _add(
        tech_section,
        "exposure_highlights_pct",
        "exposure_clipped_highlights_pct",
        tech.exposure_clipped_highlights_pct,
        "% pixels clipped at 255; <2=acceptable, 2–5=caution, >5=overexposed",
    )
    _add(
        tech_section,
        "exposure_shadows_pct",
        "exposure_clipped_shadows_pct",
        tech.exposure_clipped_shadows_pct,
        "% pixels clipped at 0; <2=acceptable, >5=underexposed",
    )
    _add(
        tech_section,
        "histogram_mean",
        "histogram_mean",
        tech.histogram_mean,
        "mean luminance 0–255; <80=dark, 80–170=well-exposed, >200=very bright",
    )
    _add(
        tech_section,
        "histogram_std",
        "histogram_std",
        tech.histogram_std,
        "luminance spread; <30=flat/low contrast, >80=wide tonal range",
    )
    _add(
        tech_section,
        "dynamic_range_stops",
        "dynamic_range_stops",
        tech.dynamic_range_stops,
        "log2(p99/p1); <3=flat/low dynamic range, 3–5=moderate, >5=rich",
    )
    _add(
        tech_section,
        "contrast_rms",
        "contrast_rms",
        tech.contrast_rms,
        "std/mean of grayscale; <0.2=flat, 0.2–0.5=normal, >0.5=punchy",
    )

    comp_section: dict = {}
    _add(
        comp_section,
        "saliency_centroid_x",
        "saliency_centroid_x",
        comp.saliency_centroid_x,
        "0=left edge, 1=right edge; 0.333 and 0.667 are rule-of-thirds lines",
    )
    _add(
        comp_section,
        "saliency_centroid_y",
        "saliency_centroid_y",
        comp.saliency_centroid_y,
        "0=top edge, 1=bottom edge; 0.333 and 0.667 are rule-of-thirds lines",
    )
    _add(
        comp_section,
        "rot_alignment_score",
        "rot_alignment_score",
        comp.rot_alignment_score,
        "0=perfectly on a RoT power point, 1=worst; <0.15=excellent",
    )
    _add(
        comp_section,
        "golden_ratio_alignment_score",
        "golden_ratio_alignment_score",
        comp.golden_ratio_alignment_score,
        "0=perfectly on a Golden Ratio power point, 1=worst; <0.15=excellent",
    )
    if is_enabled("best_alignment"):
        comp_section["best_alignment"] = {
            "value": comp.best_alignment,
            "scale": "rule_of_thirds | golden_ratio — which framework the subject aligns to",
        }
    _add(
        comp_section,
        "negative_space_ratio",
        "negative_space_ratio",
        comp.negative_space_ratio,
        "0–1 fraction of non-salient pixels; >0.5=minimalist composition",
    )
    if is_enabled("visual_weight_quadrants") and comp.visual_weight_quadrants:
        comp_section["visual_weight_quadrants"] = {
            "value": comp.visual_weight_quadrants,
            "scale": "per-quadrant saliency fraction; sums to 1.0",
        }
    _add(
        comp_section,
        "visual_weight_balance",
        "visual_weight_balance",
        comp.visual_weight_balance,
        "heaviest/lightest quadrant ratio; 1.0=balanced, >3=heavily concentrated",
    )
    _add(
        comp_section,
        "symmetry_horizontal",
        "symmetry_horizontal",
        comp.symmetry_horizontal,
        "0–1 NCC of left vs right halves; 1.0=perfectly left-right symmetric",
    )
    _add(
        comp_section,
        "symmetry_vertical",
        "symmetry_vertical",
        comp.symmetry_vertical,
        "0–1 NCC of top vs bottom halves; 1.0=perfectly top-bottom symmetric",
    )
    if is_enabled("leading_lines_converge_to_subject"):
        comp_section["leading_lines_converge_to_subject"] = {
            "value": comp.leading_lines_converge_to_subject,
            "scale": "bool; True=detected lines guide the eye toward the subject",
        }
    if is_enabled("line_pattern"):
        comp_section["line_pattern"] = {
            "value": comp.line_pattern,
            "scale": "diagonal=dynamic, horizontal=calm/stable, vertical=formal, mixed, none",
        }
    if is_enabled("dominant_line_angles") and comp.dominant_line_angles:
        comp_section["dominant_line_angles"] = {
            "value": comp.dominant_line_angles,
            "scale": "line angles in degrees 0–180; 0/180=horizontal, 90=vertical",
        }
    _add(
        comp_section,
        "horizon_tilt_degrees",
        "horizon_tilt_degrees",
        comp.horizon_tilt_degrees,
        "degrees from level; 0=perfectly level, positive=clockwise tilt, null=no clear horizon detected",
    )
    if is_enabled("scene_type"):
        comp_section["scene_type"] = {
            "value": comp.scene_type,
            "scale": "portrait|landscape|architecture|macro|general",
        }
    if is_enabled("color_harmony_type"):
        comp_section["color_harmony_type"] = {
            "value": comp.color_harmony_type,
            "scale": "monochromatic|analogous|complementary|triadic|split-complementary|complex",
        }
    _add(
        comp_section,
        "color_harmony_score",
        "color_harmony_score",
        comp.color_harmony_score,
        "0–1, higher = stronger match to the named template",
    )

    payload = {
        "exif": exif.model_dump(exclude_none=True),
        "quality_tier": _compute_quality_tier(tech, comp),
        "technical": tech_section,
        "composition": comp_section,
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
        logger.info(
            "LLM response  finish_reason=%s  response=%d chars", finish_reason, len(raw_text)
        )
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
