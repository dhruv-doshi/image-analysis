#!/usr/bin/env python3
"""Run the full FrameIQ pipeline on a local image and print detailed diagnostics.

Usage:
    source .venv/bin/activate
    python scripts/analyse.py path/to/photo.jpg
    python scripts/analyse.py path/to/photo.jpg --debug       # per-metric timing
    python scripts/analyse.py path/to/photo.jpg --no-llm      # skip LLM call
    python scripts/analyse.py path/to/photo.jpg --payload     # print LLM input JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


logger = logging.getLogger("frameiq.analyse")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full FrameIQ pipeline on a local image"
    )
    parser.add_argument("image", type=Path, help="Path to JPEG or PNG image")
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging (per-metric timing)"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Skip the LLM call (faster, offline)"
    )
    parser.add_argument(
        "--payload", action="store_true", help="Print the full LLM input JSON payload"
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
        elapsed_total = time.perf_counter() - t_total
        logger.info("--no-llm: skipping LLM call")
        logger.info("total (no LLM)  %.2fs", elapsed_total)
        return

    # ── Layer 3: LLM ──────────────────────────────────────────────────────────
    _hr("LAYER 3 — LLM")
    from src.llm.client import _MODEL, synthesise_stream
    from src.llm.synthesizer import _SYSTEM_PROMPT, _build_payload, _compute_quality_tier
    from src.models import AnalysisFeature

    quality_tier = _compute_quality_tier(tech)
    payload = _build_payload(tech, comp, exif, AnalysisFeature.FULL)
    logger.info(
        "model=%s  system_prompt=%d chars  user_payload=%d chars  max_tokens=4096",
        _MODEL,
        len(_SYSTEM_PROMPT),
        len(payload),
    )
    logger.info("quality_tier: %s", json.dumps(quality_tier))

    if args.payload:
        _hr("LLM INPUT PAYLOAD")
        print(payload)

    logger.info("calling LLM…")
    t = time.perf_counter()
    chunks: list[str] = []
    try:
        for chunk in synthesise_stream(tech, comp, exif, AnalysisFeature.FULL):
            chunks.append(chunk)
    except Exception as exc:
        logger.error("LLM stream error: %s", exc)
        sys.exit(1)

    elapsed3 = time.perf_counter() - t
    raw = "".join(chunks)
    logger.info(
        "LLM complete  %.2fs  chunks=%d  chars=%d", elapsed3, len(chunks), len(raw)
    )

    # ── Raw LLM response ──────────────────────────────────────────────────────
    _hr("LLM RAW RESPONSE")
    print(raw[:3000])
    if len(raw) > 3000:
        print(f"\n… [{len(raw) - 3000} more chars truncated]")

    # ── Parse report ──────────────────────────────────────────────────────────
    _hr("PARSED REPORT")
    from src.llm.synthesizer import _sanitise_llm_json
    buf = _sanitise_llm_json(raw)

    try:
        report = json.loads(buf)
        print(json.dumps(report, indent=2))
        logger.info("JSON parse: OK  keys=%s", list(report.keys()))
    except json.JSONDecodeError as exc:
        logger.error("JSON parse FAILED: %s", exc)
        logger.error("offending char at position %d", exc.pos)
        logger.error("context: …%s…", raw[max(0, exc.pos - 40): exc.pos + 40])
        sys.exit(1)

    # ── Timing summary ────────────────────────────────────────────────────────
    _hr("TIMING SUMMARY")
    elapsed_total = time.perf_counter() - t_total
    logger.info("L1 technical:   %.2fs", elapsed1)
    logger.info("L2 composition: %.2fs", elapsed2)
    logger.info("L3 LLM:         %.2fs", elapsed3)
    logger.info("total:          %.2fs", elapsed_total)


if __name__ == "__main__":
    main()
