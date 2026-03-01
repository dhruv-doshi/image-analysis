"""
bias_detector.py — Combine numerical vs inferred tier to produce bias statistics.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from tests.evaluation.pipeline_runner import PipelineResult
from tests.evaluation.score_classifier import (
    TIER_AVERAGE,
    TIER_POOR,
    TIER_TERRIBLE,
    QualityTier,
    derive_numerical_tier,
    flag_poor_metrics,
)
from tests.evaluation.text_analyzer import TextAnalysis


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BiasResult:
    image_id: str
    expected_tier: QualityTier | None
    inferred_tier: QualityTier | None
    numerical_tier: QualityTier
    tier_gap: int            # inferred.ordinal - numerical.ordinal; positive = Claude inflated
    tier_mismatch: bool      # abs(tier_gap) >= 1
    positivity_score: float
    coverage_rate: float     # poor metrics Claude mentioned / total poor metrics


@dataclass
class AggregateStats:
    n_images: int
    mean_positivity_score: float
    median_positivity_score: float
    false_positive_rate: float          # poor/terrible images Claude rated average or better
    mean_tier_gap: float
    worst_glossed_metrics: list[str]    # most often in glossed_over across all images
    best_reflected_metrics: list[str]   # most often in mentioned_critically
    per_tier_mean_positivity: dict[str, float]


# ---------------------------------------------------------------------------
# Per-image bias computation
# ---------------------------------------------------------------------------

def compute_bias_result(
    pipeline_result: PipelineResult,
    text_analysis: TextAnalysis,
) -> BiasResult:
    """Combine pipeline metrics and text analysis into a single BiasResult."""
    tech = pipeline_result.tech
    comp = pipeline_result.comp

    # Should not happen if called correctly, but guard defensively
    if tech is None or comp is None:
        raise ValueError(
            f"Cannot compute bias for {pipeline_result.image_id}: "
            "Layer 1 or Layer 2 scores are missing."
        )

    numerical_tier = derive_numerical_tier(tech)
    inferred_tier = text_analysis.inferred_tier

    if inferred_tier is not None:
        tier_gap = inferred_tier.ordinal - numerical_tier.ordinal
    else:
        tier_gap = 0

    poor_metrics = flag_poor_metrics(tech, comp)
    total_poor = len(poor_metrics)
    mentioned_count = len(text_analysis.poor_metrics_mentioned_critically)
    coverage_rate = (mentioned_count / total_poor) if total_poor > 0 else 1.0

    return BiasResult(
        image_id=pipeline_result.image_id,
        expected_tier=pipeline_result.expected_tier,
        inferred_tier=inferred_tier,
        numerical_tier=numerical_tier,
        tier_gap=tier_gap,
        tier_mismatch=abs(tier_gap) >= 1,
        positivity_score=text_analysis.positivity_score,
        coverage_rate=coverage_rate,
    )


def is_false_positive(bias_result: BiasResult) -> bool:
    """True when numerical tier is poor/terrible but Claude rated average or better."""
    if bias_result.numerical_tier.ordinal < TIER_POOR.ordinal:
        return False
    if bias_result.inferred_tier is None:
        return False
    return bias_result.inferred_tier.ordinal <= TIER_AVERAGE.ordinal


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(
    bias_results: list[BiasResult],
    text_analyses: list[TextAnalysis],
) -> AggregateStats:
    """Compute aggregate statistics across all images.

    Args:
        bias_results:   One BiasResult per image with LLM output.
        text_analyses:  Matching TextAnalysis list (same order / image_id).
    """
    n = len(bias_results)
    if n == 0:
        return AggregateStats(
            n_images=0,
            mean_positivity_score=0.0,
            median_positivity_score=0.0,
            false_positive_rate=0.0,
            mean_tier_gap=0.0,
            worst_glossed_metrics=[],
            best_reflected_metrics=[],
            per_tier_mean_positivity={},
        )

    positivity_scores = [r.positivity_score for r in bias_results]
    mean_pos = statistics.mean(positivity_scores)
    median_pos = statistics.median(positivity_scores)

    # False positive rate: poor/terrible images rated average-or-better by Claude
    poor_terrible = [
        r for r in bias_results
        if r.numerical_tier.ordinal >= TIER_POOR.ordinal
    ]
    fp_count = sum(1 for r in poor_terrible if is_false_positive(r))
    fp_rate = (fp_count / len(poor_terrible)) if poor_terrible else 0.0

    valid_gaps = [r.tier_gap for r in bias_results if r.inferred_tier is not None]
    mean_gap = statistics.mean(valid_gaps) if valid_gaps else 0.0

    # Metric coverage across all images
    glossed_counter: Counter[str] = Counter()
    mentioned_counter: Counter[str] = Counter()
    for ta in text_analyses:
        glossed_counter.update(ta.poor_metrics_glossed_over)
        mentioned_counter.update(ta.poor_metrics_mentioned_critically)

    worst_glossed = [m for m, _ in glossed_counter.most_common(5)]
    best_reflected = [m for m, _ in mentioned_counter.most_common(5)]

    # Positivity by numerical tier
    tier_groups: dict[str, list[float]] = {}
    for r in bias_results:
        label = r.numerical_tier.label
        tier_groups.setdefault(label, []).append(r.positivity_score)
    per_tier_mean = {
        label: statistics.mean(scores) for label, scores in tier_groups.items()
    }

    return AggregateStats(
        n_images=n,
        mean_positivity_score=mean_pos,
        median_positivity_score=median_pos,
        false_positive_rate=fp_rate,
        mean_tier_gap=mean_gap,
        worst_glossed_metrics=worst_glossed,
        best_reflected_metrics=best_reflected,
        per_tier_mean_positivity=per_tier_mean,
    )


# ---------------------------------------------------------------------------
# Prompt improvement suggestions (rule-based)
# ---------------------------------------------------------------------------

def generate_prompt_suggestions(stats: AggregateStats) -> list[str]:
    """Generate concrete system-prompt improvement suggestions based on stats."""
    suggestions: list[str] = []

    if stats.false_positive_rate > 0.3:
        suggestions.append(
            f"High false-positive rate ({stats.false_positive_rate:.0%}): Add an explicit "
            "tier instruction to system.md — e.g., 'When BRISQUE > 65 or sharpness < 200, "
            "you MUST identify the image as poor quality in the technical section.'"
        )

    if stats.mean_tier_gap > 0.75:
        suggestions.append(
            f"Mean tier inflation of {stats.mean_tier_gap:.2f} tiers: Instruct Claude to "
            "anchor its qualitative summary to the numerical scores provided — "
            "e.g., 'Your summary must reflect the numerical quality tier; do not upgrade "
            "the tone if scores indicate poor quality.'"
        )

    if stats.mean_positivity_score > 0.7:
        suggestions.append(
            f"Mean positivity score is {stats.mean_positivity_score:.2f} (threshold 0.70): "
            "Add a critical-feedback requirement — "
            "e.g., 'For every significant technical flaw identified, use direct language "
            "(blurry, overexposed, noisy) rather than softening euphemisms.'"
        )

    for metric in stats.worst_glossed_metrics[:3]:
        human_name = {
            "brisque": "BRISQUE perceptual quality",
            "sharpness_laplacian": "sharpness (Laplacian variance)",
            "noise_sigma": "noise level",
            "exposure_highlights": "highlight clipping",
            "exposure_shadows": "shadow clipping",
            "histogram_mean": "exposure/luminance",
            "visual_weight_balance": "visual weight balance",
            "rot_alignment_score": "rule-of-thirds alignment",
        }.get(metric, metric)
        suggestions.append(
            f"Metric '{human_name}' frequently glossed over: Add an explicit threshold "
            f"anchor in system.md — e.g., 'When {human_name} is in the poor range, you "
            f"MUST mention it by name in the technical section.'"
        )

    if not suggestions:
        suggestions.append(
            "No significant bias detected. Claude appears to reflect the numerical scores "
            "accurately. Consider running with a larger or more diverse image set to confirm."
        )

    return suggestions
