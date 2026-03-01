"""
reporter.py — Serialise evaluation results to JSON and Markdown.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from tests.evaluation.bias_detector import AggregateStats, BiasResult
from tests.evaluation.pipeline_runner import PipelineResult
from tests.evaluation.score_classifier import QualityTier, derive_numerical_tier
from tests.evaluation.text_analyzer import TextAnalysis


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _safe_value(v: object) -> object:
    """Convert non-JSON-serialisable values to their closest JSON equivalent."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, QualityTier):
        return {"label": v.label, "ordinal": v.ordinal}
    if isinstance(v, dict):
        return {k: _safe_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_safe_value(item) for item in v]
    return v


def results_to_dict(
    pipeline_results: list[PipelineResult],
    text_analyses: list[TextAnalysis],
    bias_results: list[BiasResult],
    stats: AggregateStats,
    suggestions: list[str],
) -> dict:
    """Convert all evaluation data to a JSON-serialisable dict."""

    def _pr_dict(pr: PipelineResult) -> dict:
        d: dict = {
            "image_id": pr.image_id,
            "source": pr.source,
            "expected_tier": _safe_value(pr.expected_tier),
            "pipeline_error": pr.pipeline_error,
        }
        if pr.tech:
            d["tech"] = _safe_value(pr.tech.model_dump())
        if pr.comp:
            d["comp"] = _safe_value(pr.comp.model_dump())
        if pr.exif:
            d["exif"] = _safe_value(pr.exif.model_dump(exclude_none=True))
        if pr.report:
            d["report"] = _safe_value(pr.report.model_dump(exclude_none=True))
        return d

    def _ta_dict(ta: TextAnalysis) -> dict:
        return {
            "image_id": ta.image_id,
            "inferred_tier": _safe_value(ta.inferred_tier),
            "positivity_score": ta.positivity_score,
            "positive_word_count": ta.positive_word_count,
            "critical_word_count": ta.critical_word_count,
            "poor_metrics_mentioned_critically": ta.poor_metrics_mentioned_critically,
            "poor_metrics_glossed_over": ta.poor_metrics_glossed_over,
        }

    def _br_dict(br: BiasResult) -> dict:
        return {
            "image_id": br.image_id,
            "expected_tier": _safe_value(br.expected_tier),
            "inferred_tier": _safe_value(br.inferred_tier),
            "numerical_tier": _safe_value(br.numerical_tier),
            "tier_gap": br.tier_gap,
            "tier_mismatch": br.tier_mismatch,
            "positivity_score": br.positivity_score,
            "coverage_rate": br.coverage_rate,
        }

    ta_by_id = {ta.image_id: ta for ta in text_analyses}
    br_by_id = {br.image_id: br for br in bias_results}

    return {
        "aggregate": {
            "n_images": stats.n_images,
            "mean_positivity_score": stats.mean_positivity_score,
            "median_positivity_score": stats.median_positivity_score,
            "false_positive_rate": stats.false_positive_rate,
            "mean_tier_gap": stats.mean_tier_gap,
            "worst_glossed_metrics": stats.worst_glossed_metrics,
            "best_reflected_metrics": stats.best_reflected_metrics,
            "per_tier_mean_positivity": stats.per_tier_mean_positivity,
        },
        "suggestions": suggestions,
        "per_image": [
            {
                "pipeline": _pr_dict(pr),
                "text_analysis": _ta_dict(ta_by_id[pr.image_id]) if pr.image_id in ta_by_id else None,
                "bias": _br_dict(br_by_id[pr.image_id]) if pr.image_id in br_by_id else None,
            }
            for pr in pipeline_results
        ],
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–1 value."""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.2f}"


def render_markdown(
    pipeline_results: list[PipelineResult],
    text_analyses: list[TextAnalysis],
    bias_results: list[BiasResult],
    stats: AggregateStats,
    suggestions: list[str],
) -> str:
    ta_by_id = {ta.image_id: ta for ta in text_analyses}
    br_by_id = {br.image_id: br for br in bias_results}

    lines: list[str] = []

    # Title
    lines.append("# FrameIQ Evaluation Report\n")

    # --- Summary stats ---
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Images analysed | {stats.n_images} |")
    lines.append(f"| Mean positivity score | {stats.mean_positivity_score:.3f} |")
    lines.append(f"| Median positivity score | {stats.median_positivity_score:.3f} |")
    lines.append(f"| False-positive rate | {stats.false_positive_rate:.1%} |")
    lines.append(f"| Mean tier gap (inflation) | {stats.mean_tier_gap:+.2f} |")
    lines.append("")

    # --- Per-image scorecard ---
    lines.append("## Per-Image Scorecard\n")
    lines.append(
        "| Image ID | Expected | Numerical | Claude Inferred | Gap | FP? "
        "| Positivity | Coverage |"
    )
    lines.append("|----------|----------|-----------|-----------------|-----|-----|------------|----------|")

    for pr in pipeline_results:
        br = br_by_id.get(pr.image_id)
        ta = ta_by_id.get(pr.image_id)

        expected = pr.expected_tier.label if pr.expected_tier else "—"
        # Compute numerical tier from tech scores directly (available even without LLM)
        if br:
            numerical = br.numerical_tier.label
        elif pr.tech:
            numerical = derive_numerical_tier(pr.tech).label
        else:
            numerical = "—"
        inferred = (br.inferred_tier.label if br and br.inferred_tier else "—") if br else "—"
        gap = f"{br.tier_gap:+d}" if br and br.inferred_tier else "—"
        fp = "YES" if br and br.inferred_tier and br.numerical_tier.ordinal >= 3 and br.inferred_tier.ordinal <= 2 else "no"
        positivity = f"{br.positivity_score:.2f}" if br else "—"
        coverage = f"{br.coverage_rate:.0%}" if br else "—"
        error = f" ⚠ {pr.pipeline_error[:40]}" if pr.pipeline_error else ""

        lines.append(
            f"| {pr.image_id}{error} | {expected} | {numerical} | {inferred} "
            f"| {gap} | {fp} | {positivity} | {coverage} |"
        )

    lines.append("")

    # --- Metric coverage ---
    lines.append("## Worst Glossed-Over Metrics\n")
    if stats.worst_glossed_metrics:
        for m in stats.worst_glossed_metrics:
            lines.append(f"- `{m}`")
    else:
        lines.append("_None — good coverage!_")
    lines.append("")

    lines.append("## Best Reflected Metrics\n")
    if stats.best_reflected_metrics:
        for m in stats.best_reflected_metrics:
            lines.append(f"- `{m}`")
    else:
        lines.append("_No poor metrics were mentioned critically._")
    lines.append("")

    # --- ASCII positivity-by-tier bar chart ---
    if stats.per_tier_mean_positivity:
        lines.append("## Positivity Score by Numerical Tier\n")
        lines.append("```")
        tier_order = ["excellent", "good", "average", "poor", "terrible"]
        for tier_label in tier_order:
            if tier_label in stats.per_tier_mean_positivity:
                score = stats.per_tier_mean_positivity[tier_label]
                lines.append(f"{tier_label:10s} {_bar(score)}")
        lines.append("```")
        lines.append(
            "\n_Higher positivity for lower-quality tiers indicates positive bias._\n"
        )

    # --- Suggestions ---
    lines.append("## Prompt Improvement Suggestions\n")
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"{i}. {suggestion}")
    lines.append("")

    # --- Methodology ---
    lines.append("## Methodology\n")
    lines.append(
        "- **Numerical tier** is computed from Layer 1+2 scores using a weighted vote "
        "(BRISQUE×3, sharpness×2, noise×2, exposure×1)."
    )
    lines.append(
        "- **Inferred tier** is extracted from Claude's AnalysisReport text by scanning "
        "for tier-signal vocabulary with simple negation handling."
    )
    lines.append(
        "- **Tier gap** = inferred.ordinal − numerical.ordinal; "
        "positive values indicate Claude inflated the quality."
    )
    lines.append(
        "- **False positive** = numerical tier is poor/terrible (ordinal ≥ 3) "
        "AND inferred tier is average or better (ordinal ≤ 2)."
    )
    lines.append(
        "- **Positivity score** = positive_word_count / (positive + critical); "
        "0.5 = neutral baseline."
    )
    lines.append(
        "- **Coverage rate** = poor metrics Claude mentioned critically / total poor metrics. "
        "1.0 = full coverage."
    )
    lines.append(
        "- Synthetic images use gradient + noise patterns; rembg falls back to uniform "
        "saliency for these (centroid fixed at 0.5, 0.5). This is expected behaviour."
    )
    lines.append("")

    return "\n".join(lines)


def save_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
