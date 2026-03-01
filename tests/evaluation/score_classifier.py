"""
score_classifier.py — Map raw pipeline scores to quality tiers and flag poor metrics.

Thresholds are aligned with the scale annotations in src/llm/synthesizer.py _build_payload().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.models import CompositionScores, TechnicalScores


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QualityTier:
    label: str    # "excellent" | "good" | "average" | "poor" | "terrible"
    ordinal: int  # 0=best … 4=worst

    def __repr__(self) -> str:
        return f"QualityTier(label={self.label!r}, ordinal={self.ordinal})"


@dataclass
class MetricFlag:
    metric_name: str           # e.g. "brisque", "sharpness_laplacian"
    value: float
    is_poor: bool
    poor_threshold_desc: str   # human-readable, e.g. "BRISQUE > 70"


# Canonical tier singletons (avoids constructing new objects everywhere)
TIER_EXCELLENT = QualityTier("excellent", 0)
TIER_GOOD      = QualityTier("good",      1)
TIER_AVERAGE   = QualityTier("average",   2)
TIER_POOR      = QualityTier("poor",      3)
TIER_TERRIBLE  = QualityTier("terrible",  4)

_ALL_TIERS = [TIER_EXCELLENT, TIER_GOOD, TIER_AVERAGE, TIER_POOR, TIER_TERRIBLE]


def _by_label(label: str) -> QualityTier:
    for t in _ALL_TIERS:
        if t.label == label:
            return t
    return TIER_AVERAGE


# ---------------------------------------------------------------------------
# Per-metric classifiers
# ---------------------------------------------------------------------------

def classify_brisque(value: float) -> QualityTier:
    """BRISQUE: 0–100, lower is better.
    <30=excellent, 30–50=good, 50–65=average, 65–80=poor, >80=terrible.
    nan → average (fallback with log-warning at call site).
    """
    if math.isnan(value):
        return TIER_AVERAGE
    if value < 30:
        return TIER_EXCELLENT
    if value < 50:
        return TIER_GOOD
    if value < 65:
        return TIER_AVERAGE
    if value < 80:
        return TIER_POOR
    return TIER_TERRIBLE


def classify_sharpness(value: float) -> QualityTier:
    """Laplacian variance: higher = sharper.
    >500=excellent, 200–500=average, <200=poor.
    """
    if math.isnan(value):
        return TIER_AVERAGE
    if value > 500:
        return TIER_EXCELLENT
    if value >= 200:
        return TIER_AVERAGE
    if value >= 80:
        return TIER_POOR
    return TIER_TERRIBLE


def classify_noise(value: float) -> QualityTier:
    """Gaussian noise sigma: lower = cleaner.
    <3=excellent, 3–8=average, >8=poor.
    """
    if math.isnan(value):
        return TIER_AVERAGE
    if value < 3:
        return TIER_EXCELLENT
    if value <= 8:
        return TIER_AVERAGE
    if value <= 15:
        return TIER_POOR
    return TIER_TERRIBLE


def classify_exposure(
    highlights_pct: float,
    shadows_pct: float,
    hist_mean: float,
) -> QualityTier:
    """Exposure quality from three complementary metrics.
    Highlights: <2=ok, 2-5=caution, >5=poor.
    Shadows:    <2=ok, 2-5=caution, >5=poor.
    Hist mean:  80-170=ok, <80=dark(poor), >200=bright(poor).
    """
    # Count how many sub-metrics are poor
    poor_count = 0
    severe_count = 0

    if highlights_pct > 5:
        poor_count += 1
    if highlights_pct > 15:
        severe_count += 1

    if shadows_pct > 5:
        poor_count += 1
    if shadows_pct > 20:
        severe_count += 1

    if hist_mean < 80 or hist_mean > 200:
        poor_count += 1
    if hist_mean < 40 or hist_mean > 230:
        severe_count += 1

    if severe_count >= 2:
        return TIER_TERRIBLE
    if severe_count >= 1:
        return TIER_POOR
    if poor_count >= 2:
        return TIER_POOR
    if poor_count >= 1:
        return TIER_AVERAGE
    # Check caution zone
    caution = 0
    if highlights_pct > 2:
        caution += 1
    if shadows_pct > 2:
        caution += 1
    if caution >= 1:
        return TIER_GOOD
    return TIER_EXCELLENT


# ---------------------------------------------------------------------------
# Aggregate numerical tier
# ---------------------------------------------------------------------------

def derive_numerical_tier(tech: TechnicalScores) -> QualityTier:
    """Weighted vote: BRISQUE×3, sharpness×2, noise×2, exposure×1.

    Sharpness is noise-adjusted before classification: the Laplacian variance
    contribution of Gaussian noise (σ² × 20, the sum of squared kernel weights)
    is subtracted before mapping to a tier.  This prevents noisy images from
    appearing artificially sharp.

    Python's built-in round() uses banker's rounding (round-half-to-even).
    We use int(x + 0.5) instead to get standard round-half-up, so that
    avg_ordinal = 2.5 maps to tier 3 ("poor") rather than tier 2 ("average").
    """
    brisque_tier = classify_brisque(tech.brisque)

    # Noise-adjusted sharpness: subtract estimated Gaussian-noise Laplacian contribution.
    # Laplacian kernel has sum-of-squares = 20; noise variance σ² maps to σ²×20 in output.
    _LAPLACIAN_KERNEL_SQSUM = 20
    noise_adjusted_sharp = max(0.0, tech.sharpness_laplacian - tech.noise_sigma ** 2 * _LAPLACIAN_KERNEL_SQSUM)
    sharp_tier    = classify_sharpness(noise_adjusted_sharp)

    noise_tier    = classify_noise(tech.noise_sigma)
    exposure_tier = classify_exposure(
        tech.exposure_clipped_highlights_pct,
        tech.exposure_clipped_shadows_pct,
        tech.histogram_mean,
    )

    weights = {"brisque": 3, "sharpness": 2, "noise": 2, "exposure": 1}
    tiers   = {
        "brisque":   brisque_tier,
        "sharpness": sharp_tier,
        "noise":     noise_tier,
        "exposure":  exposure_tier,
    }

    total_weight    = sum(weights.values())   # 8
    weighted_sum    = sum(tiers[k].ordinal * w for k, w in weights.items())
    avg_ordinal     = weighted_sum / total_weight

    # Standard round-half-up (avoids banker's rounding artefacts at x.5 boundaries)
    rounded = int(avg_ordinal + 0.5)
    rounded = max(0, min(4, rounded))
    return _ALL_TIERS[rounded]


# ---------------------------------------------------------------------------
# Flag individual poor metrics
# ---------------------------------------------------------------------------

def flag_poor_metrics(
    tech: TechnicalScores,
    comp: CompositionScores,
) -> list[MetricFlag]:
    """Return a list of MetricFlag for every metric in the 'poor' range."""
    flags: list[MetricFlag] = []

    def _flag(name: str, value: float, is_poor: bool, desc: str) -> None:
        flags.append(MetricFlag(
            metric_name=name,
            value=value,
            is_poor=is_poor,
            poor_threshold_desc=desc,
        ))

    # BRISQUE
    brisque_poor = not math.isnan(tech.brisque) and tech.brisque >= 65
    _flag("brisque", tech.brisque, brisque_poor, "BRISQUE >= 65")

    # Sharpness
    sharp_poor = tech.sharpness_laplacian < 200
    _flag("sharpness_laplacian", tech.sharpness_laplacian, sharp_poor, "Laplacian variance < 200")

    # Noise
    noise_poor = tech.noise_sigma > 8
    _flag("noise_sigma", tech.noise_sigma, noise_poor, "Noise sigma > 8")

    # Highlights
    hl_poor = tech.exposure_clipped_highlights_pct > 5
    _flag(
        "exposure_highlights",
        tech.exposure_clipped_highlights_pct,
        hl_poor,
        "Clipped highlights > 5%",
    )

    # Shadows
    sh_poor = tech.exposure_clipped_shadows_pct > 5
    _flag(
        "exposure_shadows",
        tech.exposure_clipped_shadows_pct,
        sh_poor,
        "Clipped shadows > 5%",
    )

    # Histogram mean
    mean_poor = tech.histogram_mean < 80 or tech.histogram_mean > 200
    _flag("histogram_mean", tech.histogram_mean, mean_poor, "Histogram mean < 80 or > 200")

    # Composition — visual weight balance
    balance_poor = comp.visual_weight_balance > 4.0
    _flag(
        "visual_weight_balance",
        comp.visual_weight_balance,
        balance_poor,
        "Visual weight balance ratio > 4.0 (heavily unbalanced)",
    )

    # Composition — RoT alignment
    rot_poor = comp.rot_alignment_score > 0.4
    _flag("rot_alignment_score", comp.rot_alignment_score, rot_poor, "RoT alignment score > 0.4")

    return [f for f in flags if f.is_poor]
