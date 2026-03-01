"""
text_analyzer.py — Parse Claude's AnalysisReport for sentiment and tier signals.

Pure string analysis — no second LLM call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models import AnalysisReport
from tests.evaluation.score_classifier import (
    TIER_AVERAGE,
    TIER_EXCELLENT,
    TIER_GOOD,
    TIER_POOR,
    TIER_TERRIBLE,
    MetricFlag,
    QualityTier,
)


# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

POSITIVE_WORDS: frozenset[str] = frozenset({
    "excellent", "sharp", "crisp", "clean", "balanced", "effective", "strong",
    "good", "great", "rich", "vibrant", "compelling", "successful", "impressive",
    "beautiful", "stunning", "lovely", "remarkable", "outstanding", "superb",
    "exceptional", "perfect", "well", "nice", "pleasing", "pleasing",
    "dynamic", "interesting", "engaging", "expressive", "intentional",
    "professional", "polished", "accomplished", "skillful", "well-exposed",
    "well-composed", "well-balanced",
})

CRITICAL_WORDS: frozenset[str] = frozenset({
    "blurry", "blur", "noisy", "noise", "grain", "grainy", "overexposed",
    "underexposed", "clipped", "flat", "poor", "weak", "lacks", "awkward",
    "unbalanced", "muddy", "dull", "washed", "harsh", "distracting",
    "cluttered", "busy", "confusing", "unclear", "inconsistent", "excessive",
    "unfortunately", "however", "although", "despite", "issue", "problem",
    "needs work", "consider", "would benefit", "could benefit", "could be improved",
    "improvement", "missing", "absent", "lacking", "insufficient", "too dark",
    "too bright", "out of focus", "soft focus", "motion blur", "camera shake",
    "overexposure", "underexposure", "blown", "crushed", "loss of detail",
    "limited", "narrow", "minimal",
})

# Maps each pipeline metric name → keywords Claude would use if it flagged it
METRIC_COVERAGE_KEYWORDS: dict[str, frozenset[str]] = {
    "brisque": frozenset({"quality", "overall quality", "image quality", "clarity"}),
    "sharpness_laplacian": frozenset({
        "blur", "blurry", "soft", "sharpness", "focus", "crisp", "sharp", "unsharp",
        "out of focus", "motion blur", "camera shake",
    }),
    "noise_sigma": frozenset({
        "noise", "noisy", "grain", "grainy", "luminance noise", "color noise",
    }),
    "exposure_highlights": frozenset({
        "overexposed", "highlights", "blown", "clipped", "brightness", "bright", "overexposure",
    }),
    "exposure_shadows": frozenset({
        "underexposed", "shadows", "dark", "darkness", "crushed", "underexposure",
        "shadow detail", "shadow clipping",
    }),
    "histogram_mean": frozenset({
        "exposure", "brightness", "luminance", "tonal", "tone", "dark", "bright",
        "well-exposed", "midtones",
    }),
    "visual_weight_balance": frozenset({
        "balance", "balanced", "unbalanced", "weight", "visual weight", "composition balance",
    }),
    "rot_alignment_score": frozenset({
        "rule of thirds", "composition", "placement", "subject placement", "thirds",
        "off-center", "centered",
    }),
}

# Tier signal words — used to infer Claude's implied quality tier
_TIER_SIGNALS: list[tuple[frozenset[str], QualityTier]] = [
    (frozenset({"exceptional", "outstanding", "stunning", "superb", "excellent",
                "exemplary", "flawless", "masterful"}), TIER_EXCELLENT),
    (frozenset({"good", "solid", "competent", "effective", "well-executed",
                "pleasing", "nice"}), TIER_GOOD),
    (frozenset({"average", "acceptable", "adequate", "decent", "moderate",
                "mixed", "some issues"}), TIER_AVERAGE),
    (frozenset({"poor", "significant issues", "major issues", "weak", "problematic",
                "needs considerable", "substantially", "seriously"}), TIER_POOR),
    (frozenset({"terrible", "unusable", "very poor", "severe", "extremely blurry",
                "completely overexposed", "completely underexposed"}), TIER_TERRIBLE),
]

# Negation words: if any of these appear within 3 words before a signal, flip it
_NEGATION_WORDS = frozenset({"not", "no", "never", "without", "lacks", "missing", "barely"})


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class TextAnalysis:
    image_id: str
    inferred_tier: QualityTier | None
    positivity_score: float                   # pos_words / (pos + crit); 0.5 if both zero
    positive_word_count: int
    critical_word_count: int
    poor_metrics_mentioned_critically: list[str] = field(default_factory=list)
    poor_metrics_glossed_over: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_text(report: AnalysisReport) -> str:
    """Concatenate all non-None fields of a report into one lowercase string."""
    parts = [
        report.summary or "",
        report.composition or "",
        report.aesthetics or "",
        report.technical or "",
        report.improvements or "",
        report.editing or "",
        report.inspiration or "",
    ]
    return " ".join(p for p in parts if p).lower()


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens (lowercase, strip punctuation)."""
    return re.findall(r"[a-z']+", text.lower())


def _has_negation_before(tokens: list[str], match_idx: int, window: int = 3) -> bool:
    """Check if any negation word appears within `window` tokens before match_idx."""
    start = max(0, match_idx - window)
    return bool(_NEGATION_WORDS.intersection(tokens[start:match_idx]))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def count_sentiment_words(text: str) -> tuple[int, int]:
    """Return (positive_count, critical_count) for the given text.

    Uses phrase matching for multi-word critical expressions first,
    then word-level matching.
    """
    text_lower = text.lower()
    pos_count = 0
    crit_count = 0

    # Multi-word critical phrases
    multi_phrases = [w for w in CRITICAL_WORDS if " " in w]
    single_critical = frozenset(w for w in CRITICAL_WORDS if " " not in w)

    for phrase in multi_phrases:
        crit_count += text_lower.count(phrase)

    tokens = _tokenize(text_lower)
    for idx, token in enumerate(tokens):
        if token in POSITIVE_WORDS and not _has_negation_before(tokens, idx):
            pos_count += 1
        elif token in single_critical:
            crit_count += 1

    return pos_count, crit_count


def infer_tier_from_text(text: str) -> QualityTier | None:
    """Scan the text for tier-signal words to infer Claude's implied quality tier.

    Checks the summary portion first (first 400 chars), then the full text.
    Handles simple negation within 3 words.
    Returns the strongest (worst) tier signal found, or None if no signal.
    """
    # Look at summary section first, then full text
    search_text = text.lower()
    tokens = _tokenize(search_text)

    tier_votes: dict[int, int] = {}  # ordinal → count

    for signal_words, tier in _TIER_SIGNALS:
        for phrase in signal_words:
            words_in_phrase = phrase.split()
            if len(words_in_phrase) == 1:
                # Single word — use token list for negation check
                for idx, tok in enumerate(tokens):
                    if tok == phrase and not _has_negation_before(tokens, idx):
                        tier_votes[tier.ordinal] = tier_votes.get(tier.ordinal, 0) + 1
            else:
                # Multi-word phrase — simple substring search
                if phrase in search_text:
                    tier_votes[tier.ordinal] = tier_votes.get(tier.ordinal, 0) + 1

    if not tier_votes:
        return None

    # Return the tier with the most votes; on tie, prefer the worse tier
    best_ordinal = max(tier_votes, key=lambda o: (tier_votes[o], o))
    tier_map = {t.ordinal: t for t in [
        TIER_EXCELLENT, TIER_GOOD, TIER_AVERAGE, TIER_POOR, TIER_TERRIBLE
    ]}
    return tier_map.get(best_ordinal)


def check_metric_coverage(
    poor_metrics: list[MetricFlag],
    report: AnalysisReport,
) -> tuple[list[str], list[str]]:
    """Determine which poor metrics Claude mentioned critically vs glossed over.

    Checks report.technical and report.improvements text only.

    Returns:
        (mentioned_critically, glossed_over)  — lists of metric_name strings.
    """
    check_text = " ".join(filter(None, [report.technical, report.improvements])).lower()

    mentioned: list[str] = []
    glossed: list[str] = []

    for flag in poor_metrics:
        keywords = METRIC_COVERAGE_KEYWORDS.get(flag.metric_name, frozenset())
        found = any(kw in check_text for kw in keywords)
        if found:
            mentioned.append(flag.metric_name)
        else:
            glossed.append(flag.metric_name)

    return mentioned, glossed


def analyze_report(
    image_id: str,
    report: AnalysisReport,
    poor_metrics: list[MetricFlag],
) -> TextAnalysis:
    """Produce a TextAnalysis from a Claude AnalysisReport.

    Args:
        image_id:     Identifier for the image.
        report:       Claude's AnalysisReport.
        poor_metrics: MetricFlags already computed by flag_poor_metrics().

    Returns:
        TextAnalysis dataclass.
    """
    full_text = _collect_text(report)

    pos_count, crit_count = count_sentiment_words(full_text)
    total = pos_count + crit_count
    positivity_score = (pos_count / total) if total > 0 else 0.5

    inferred = infer_tier_from_text(full_text)
    mentioned, glossed = check_metric_coverage(poor_metrics, report)

    return TextAnalysis(
        image_id=image_id,
        inferred_tier=inferred,
        positivity_score=positivity_score,
        positive_word_count=pos_count,
        critical_word_count=crit_count,
        poor_metrics_mentioned_critically=mentioned,
        poor_metrics_glossed_over=glossed,
    )
