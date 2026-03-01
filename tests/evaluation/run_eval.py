"""
run_eval.py — CLI entry point for the FrameIQ pipeline evaluation framework.

Usage examples:
    # Layer 1+2 only (no API key needed)
    python tests/evaluation/run_eval.py --no-llm

    # Full pipeline including Claude
    python tests/evaluation/run_eval.py

    # With user photos
    python tests/evaluation/run_eval.py --user-images tests/evaluation/images/

    # Only specific tiers
    python tests/evaluation/run_eval.py --tier poor --tier terrible

    # Verbose output
    python tests/evaluation/run_eval.py --verbose

    # Custom output location
    python tests/evaluation/run_eval.py --output-dir tests/evaluation/results/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is importable regardless of CWD
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FrameIQ Pipeline Evaluation — bias detection and score consistency check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Layer 3 (Claude API). Runs Layer 1+2 only. No API key needed.",
    )
    parser.add_argument(
        "--user-images",
        metavar="DIR",
        help="Directory of user-supplied JPEG/PNG images to include alongside synthetic images.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        metavar="TIER",
        dest="tiers",
        choices=["excellent", "good", "average", "poor", "terrible"],
        help="Only run specific synthetic tiers (repeatable). Default: all five.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image progress and scores to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="tests/evaluation/results",
        help="Directory for results.json and report.md (default: tests/evaluation/results/).",
    )
    return parser.parse_args()


def main() -> None:
    _load_env()
    args = _parse_args()

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s | %(name)s | %(message)s",
        level=log_level,
        stream=sys.stderr,
    )
    logger = logging.getLogger("run_eval")

    # -----------------------------------------------------------------------
    # API key check
    # -----------------------------------------------------------------------
    run_llm = not args.no_llm
    if run_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "WARNING: ANTHROPIC_API_KEY not set. Downgrading to --no-llm mode.\n"
            "         Set the key in .env or export ANTHROPIC_API_KEY=... to enable Claude.",
            file=sys.stderr,
        )
        run_llm = False

    # -----------------------------------------------------------------------
    # Imports (deferred so --help is instant)
    # -----------------------------------------------------------------------
    from tests.evaluation.bias_detector import (
        compute_aggregate_stats,
        compute_bias_result,
        generate_prompt_suggestions,
    )
    from tests.evaluation.image_factory import generate_tier_images, load_user_images
    from tests.evaluation.pipeline_runner import PipelineResult, run_pipeline
    from tests.evaluation.reporter import (
        render_markdown,
        results_to_dict,
        save_json,
        save_markdown,
    )
    from tests.evaluation.score_classifier import QualityTier, flag_poor_metrics
    from tests.evaluation.text_analyzer import TextAnalysis, analyze_report

    # -----------------------------------------------------------------------
    # Build image list
    # -----------------------------------------------------------------------
    tier_filter = args.tiers  # None means all five

    print("Generating synthetic images...", flush=True)
    synthetic_specs = generate_tier_images(tier_filter=tier_filter)

    images: list[tuple[str, object, QualityTier | None, str]] = []

    for spec in synthetic_specs:
        tier = QualityTier(label=spec.tier_label, ordinal={
            "excellent": 0, "good": 1, "average": 2, "poor": 3, "terrible": 4
        }[spec.tier_label])
        images.append((spec.image_id, spec.pil_image, tier, "synthetic"))

    if args.user_images:
        print(f"Loading user images from {args.user_images}...", flush=True)
        user_imgs = load_user_images(args.user_images)
        for image_id, pil_image in user_imgs:
            images.append((image_id, pil_image, None, "user"))

    if not images:
        print("No images to process. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning pipeline on {len(images)} image(s)...\n", flush=True)

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------
    pipeline_results: list[PipelineResult] = run_pipeline(
        images,
        run_llm=run_llm,
        verbose=args.verbose,
    )

    # -----------------------------------------------------------------------
    # Text analysis + bias detection
    # -----------------------------------------------------------------------
    text_analyses: list[TextAnalysis] = []
    bias_results_list = []

    for pr in pipeline_results:
        if pr.report is None or pr.tech is None or pr.comp is None:
            continue
        poor_metrics = flag_poor_metrics(pr.tech, pr.comp)
        ta = analyze_report(pr.image_id, pr.report, poor_metrics)
        text_analyses.append(ta)
        try:
            br = compute_bias_result(pr, ta)
            bias_results_list.append(br)
        except ValueError as exc:
            logger.warning("Bias computation skipped for %s: %s", pr.image_id, exc)

    # -----------------------------------------------------------------------
    # Aggregate stats + suggestions
    # -----------------------------------------------------------------------
    stats = compute_aggregate_stats(bias_results_list, text_analyses)
    suggestions = generate_prompt_suggestions(stats)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    json_path = output_dir / "results.json"
    md_path = output_dir / "report.md"

    data = results_to_dict(
        pipeline_results, text_analyses, bias_results_list, stats, suggestions
    )
    save_json(data, json_path)
    md_content = render_markdown(
        pipeline_results, text_analyses, bias_results_list, stats, suggestions
    )
    save_markdown(md_content, md_path)

    # -----------------------------------------------------------------------
    # Print summary to stdout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Images analysed:        {stats.n_images}")
    if stats.n_images > 0:
        print(f"  Mean positivity score:  {stats.mean_positivity_score:.3f}")
        print(f"  False-positive rate:    {stats.false_positive_rate:.1%}")
        print(f"  Mean tier gap:          {stats.mean_tier_gap:+.2f}")
    else:
        print("  (No LLM results to aggregate — Layer 1+2 only or all images errored)")
    print(f"\n  Results JSON:  {json_path}")
    print(f"  Report MD:     {md_path}")

    if suggestions:
        print("\nTop suggestion:")
        print(f"  → {suggestions[0]}")

    if args.verbose and pipeline_results:
        print("\n--- Per-image numerical tiers ---")
        from tests.evaluation.score_classifier import derive_numerical_tier
        for pr in pipeline_results:
            if pr.tech:
                nt = derive_numerical_tier(pr.tech)
                br = next((b for b in bias_results_list if b.image_id == pr.image_id), None)
                inferred = br.inferred_tier.label if br and br.inferred_tier else "—"
                gap = f"{br.tier_gap:+d}" if br and br.inferred_tier else "—"
                print(
                    f"  {pr.image_id:30s}  numerical={nt.label:10s}"
                    f"  inferred={inferred:10s}  gap={gap}"
                )
            else:
                print(f"  {pr.image_id:30s}  ERROR: {pr.pipeline_error}")


if __name__ == "__main__":
    main()
