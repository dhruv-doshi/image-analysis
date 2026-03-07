from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class MetricConfig(TypedDict):
    enabled: bool
    weight: float  # 0 = informational / not used in quality score


METRICS: dict[str, MetricConfig] = {
    # Technical: learned IQA
    "brisque": {"enabled": True, "weight": 3.0},
    "nima_aesthetic": {"enabled": True, "weight": 2.0},
    "clip_iqa": {"enabled": True, "weight": 1.0},
    "musiq": {"enabled": True, "weight": 1.5},
    "niqe": {"enabled": True, "weight": 1.0},
    # Technical: classical CV
    "sharpness_laplacian": {"enabled": True, "weight": 2.0},
    "sharpness_regional": {"enabled": True, "weight": 0.0},
    "noise_sigma": {"enabled": True, "weight": 2.0},
    "exposure_clipped_highlights_pct": {"enabled": True, "weight": 1.0},
    "exposure_clipped_shadows_pct": {"enabled": True, "weight": 1.0},
    "histogram_mean": {"enabled": True, "weight": 0.0},
    "histogram_std": {"enabled": True, "weight": 0.0},
    "dynamic_range_stops": {"enabled": True, "weight": 0.0},
    "contrast_rms": {"enabled": True, "weight": 0.0},
    # Composition
    "use_heavy_saliency": {"enabled": False, "weight": 0.0},  # True = rembg U²-Net, False = OpenCV spectral residual
    "saliency_centroid_x": {"enabled": True, "weight": 0.0},
    "saliency_centroid_y": {"enabled": True, "weight": 0.0},
    "rot_alignment_score": {"enabled": True, "weight": 2.0},
    "golden_ratio_alignment_score": {"enabled": True, "weight": 0.0},
    "best_alignment": {"enabled": True, "weight": 0.0},
    "negative_space_ratio": {"enabled": True, "weight": 0.0},
    "visual_weight_quadrants": {"enabled": True, "weight": 0.0},
    "visual_weight_balance": {"enabled": True, "weight": 0.0},
    "symmetry_horizontal": {"enabled": True, "weight": 0.0},
    "symmetry_vertical": {"enabled": True, "weight": 0.0},
    "dominant_line_angles": {"enabled": True, "weight": 0.0},
    "leading_lines_converge_to_subject": {"enabled": True, "weight": 0.0},
    "line_pattern": {"enabled": True, "weight": 0.0},
    "horizon_tilt_degrees": {"enabled": True, "weight": 0.0},
    "scene_type": {"enabled": True, "weight": 0.0},
    "dominant_colors": {"enabled": True, "weight": 0.0},
    "color_harmony_type": {"enabled": True, "weight": 0.0},
    "color_harmony_score": {"enabled": True, "weight": 0.0},
}

_FALLBACK: MetricConfig = {"enabled": True, "weight": 0.0}


def is_enabled(metric: str) -> bool:
    return METRICS.get(metric, _FALLBACK)["enabled"]


def get_weight(metric: str) -> float:
    return METRICS.get(metric, _FALLBACK)["weight"]


_TECH_KEYS = {
    "brisque",
    "nima_aesthetic",
    "clip_iqa",
    "musiq",
    "niqe",
    "sharpness_laplacian",
    "sharpness_regional",
    "noise_sigma",
    "exposure_clipped_highlights_pct",
    "exposure_clipped_shadows_pct",
    "histogram_mean",
    "histogram_std",
    "dynamic_range_stops",
    "contrast_rms",
}


def log_active_pipeline() -> None:
    """Log a startup summary of every metric: enabled/disabled and quality weight."""
    enabled = {k: v for k, v in METRICS.items() if v["enabled"]}
    disabled = {k: v for k, v in METRICS.items() if not v["enabled"]}

    sep = "=" * 62
    lines = [
        "",
        sep,
        "  FrameIQ -- Active Pipeline Configuration",
        sep,
        "\n  Layer 1 -- Technical Metrics",
        "  " + "-" * 42,
    ]
    for k in (k for k in METRICS if k in _TECH_KEYS):
        cfg = METRICS[k]
        status = "ON " if cfg["enabled"] else "OFF"
        wt = f"  weight={cfg['weight']:.1f}" if cfg["enabled"] and cfg["weight"] > 0 else ""
        lines.append(f"    [{status}] {k}{wt}")

    lines += ["\n  Layer 2 -- Composition Metrics", "  " + "-" * 42]
    for k in (k for k in METRICS if k not in _TECH_KEYS):
        cfg = METRICS[k]
        status = "ON " if cfg["enabled"] else "OFF"
        wt = f"  weight={cfg['weight']:.1f}" if cfg["enabled"] and cfg["weight"] > 0 else ""
        lines.append(f"    [{status}] {k}{wt}")

    lines += [
        "\n  Layer 3 -- LLM Synthesis",
        "  " + "-" * 42,
        "    [ON ] LLM call (always active)",
        "",
        f"  Enabled : {len(enabled)}/{len(METRICS)} metrics",
    ]
    if disabled:
        lines.append(f"  Disabled: {', '.join(disabled)}")
    total_w = sum(v["weight"] for v in enabled.values())
    lines.append(f"  Quality-tier weight pool: {total_w:.1f}")
    lines.append(sep)

    for line in lines:
        logger.info(line)
