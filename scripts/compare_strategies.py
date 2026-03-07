#!/usr/bin/env python3
"""Compare three LLM input strategies for photo critique on a local image.

Strategies:
  1. metrics_only      — JSON payload (no image)
  2. metrics_and_photo — JSON payload + image
  3. photo_only        — image alone (no metrics)

L1 and L2 are computed once, then all three LLM calls run in parallel via
ThreadPoolExecutor (inside run_comparison).

Usage:
    source .venv/bin/activate
    python scripts/compare_strategies.py path/to/photo.jpg
    python scripts/compare_strategies.py path/to/photo.jpg --no-llm   # L1+L2 only
    python scripts/compare_strategies.py path/to/photo.jpg --debug    # verbose logging
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-24s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpcore", "httpx", "openai", "rembg", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _hr(title: str = "") -> None:
    w = 64
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, w - len(title) - 5)}")
    else:
        print("─" * w)


logger = logging.getLogger("frameiq.compare_strategies")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three LLM input strategies on a local image"
    )
    parser.add_argument("image", type=Path, help="Path to JPEG or PNG image")
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging (verbose)"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Skip all LLM calls (L1+L2 only)"
    )
    args = parser.parse_args()

    _setup_logging(args.debug)

    if not args.image.exists():
        logger.error("File not found: %s", args.image)
        sys.exit(1)

    size_kb = args.image.stat().st_size / 1024
    print(f"\nAnalysing: {args.image}  ({size_kb:.1f} KB)")
    t_total = time.perf_counter()

    # ── Load image ────────────────────────────────────────────────────────────
    _hr("LOAD")
    from src.utils.loader import extract_exif, load_image

    t = time.perf_counter()
    pil_image, bgr_array, tensor = load_image(args.image)
    exif = extract_exif(pil_image)
    logger.info(
        "load complete  %.2fs  shape=%s  (analysis res, not original)",
        time.perf_counter() - t,
        bgr_array.shape,
    )
    exif_data = exif.model_dump(exclude_none=True)
    if exif_data:
        logger.info("exif: %s", json.dumps(exif_data))
    else:
        logger.info("exif: (none found)")

    # ── Layer 1: Technical ────────────────────────────────────────────────────
    _hr("LAYER 1 — TECHNICAL")
    from src.analysis.technical import analyse as analyse_technical

    t = time.perf_counter()
    tech = analyse_technical(bgr_array, tensor)
    elapsed1 = time.perf_counter() - t
    logger.info("L1 complete  %.2fs", elapsed1)
    print(json.dumps(tech.model_dump(), indent=2))

    # ── Layer 2: Composition ──────────────────────────────────────────────────
    _hr("LAYER 2 — COMPOSITION")
    from src.analysis.composition import analyse as analyse_composition

    t = time.perf_counter()
    comp = analyse_composition(bgr_array, pil_image, exif)
    elapsed2 = time.perf_counter() - t
    logger.info("L2 complete  %.2fs  scene=%s", elapsed2, comp.scene_type)
    comp_dump = comp.model_dump()
    comp_dump.pop("saliency_map", None)
    print(json.dumps(comp_dump, indent=2))

    if args.no_llm:
        _hr()
        logger.info("--no-llm: skipping LLM calls")
        logger.info("total (no LLM)  %.2fs", time.perf_counter() - t_total)
        return

    # ── Layer 3: LLM Comparison ───────────────────────────────────────────────
    _hr("LAYER 3 — LLM COMPARISON (3 strategies in parallel)")
    from src.llm.comparison import run_comparison

    logger.info("Firing all 3 strategies in parallel…")
    t3 = time.perf_counter()
    try:
        comparison = run_comparison(tech, comp, exif, pil_image)
    except Exception as exc:
        logger.error("Comparison failed: %s", exc)
        sys.exit(1)
    elapsed3 = time.perf_counter() - t3
    logger.info("LLM comparison complete  %.2fs", elapsed3)

    # ── Print results ─────────────────────────────────────────────────────────
    _hr("METRICS ONLY")
    print(json.dumps(comparison["metrics_only"], indent=2))

    _hr("METRICS + PHOTO")
    print(json.dumps(comparison["metrics_and_photo"], indent=2))

    _hr("PHOTO ONLY")
    print(json.dumps(comparison["photo_only"], indent=2))

    # ── Save to disk ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "prompts" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_{args.image.stem}.json"
    payload = {
        "image": str(args.image),
        "exif": exif.model_dump(),
        "technical": tech.model_dump(),
        "composition": comp_dump,
        "comparison": comparison,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _hr(f"Saved → {out_path.relative_to(Path(__file__).parent.parent)}")

    # ── Timing summary ────────────────────────────────────────────────────────
    _hr("TIMING SUMMARY")
    logger.info("L1 technical:   %.2fs", elapsed1)
    logger.info("L2 composition: %.2fs", elapsed2)
    logger.info("L3 LLM (wall):  %.2fs  (3 strategies in parallel)", elapsed3)
    logger.info("total:          %.2fs", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
