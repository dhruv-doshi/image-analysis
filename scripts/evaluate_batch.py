#!/usr/bin/env python3
"""Compare old vs new FrameIQ pipeline on a batch of images.

"Old pipeline": 12 technical + 12 composition fields (no MUSIQ, no scene context)
"New pipeline": 13 technical + 17 composition fields (full scene/color/horizon context)

Both calls use the current system prompt and the same L1/L2 analysis.
The old payload simply has the 5 new-metric fields stripped before the LLM call.

Usage:
    source .venv/bin/activate
    python scripts/evaluate_batch.py                       # all images in default dir
    python scripts/evaluate_batch.py --dir path/to/images  # custom directory
    python scripts/evaluate_batch.py --no-llm              # CV metrics only (fast)
    python scripts/evaluate_batch.py --debug               # verbose per-metric logging
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_IMAGE_DIR = Path(__file__).parent.parent / "tests" / "evaluation" / "images"
RESULTS_DIR = Path(__file__).parent.parent / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Fields added in dev-branch; absent from "old pipeline" payload
_NEW_TECH_FIELDS = {"musiq"}
_NEW_COMP_FIELDS = {
    "horizon_tilt_degrees",
    "scene_type",
    "dominant_colors",
    "color_harmony_type",
    "color_harmony_score",
}
_NEW_TIER_FIELDS = {"nima_tier", "clip_tier"}

logger = logging.getLogger("frameiq.batch")


def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-28s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpcore", "httpx", "openai", "rembg", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _strip_old_payload(payload_str: str) -> str:
    """Remove new-pipeline fields from a payload JSON string to simulate old pipeline."""
    data = json.loads(payload_str)
    for f in _NEW_TECH_FIELDS:
        data.get("technical", {}).pop(f, None)
    for f in _NEW_COMP_FIELDS:
        data.get("composition", {}).pop(f, None)
    for f in _NEW_TIER_FIELDS:
        data.get("quality_tier", {}).pop(f, None)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _call_llm(payload: str, system_prompt: str, model: str, client) -> tuple[str, dict]:
    """Call the LLM synchronously; return (raw_text, parsed_report_dict)."""
    from src.llm.synthesizer import _coerce_report_fields, _sanitise_llm_json

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
    )
    raw_text = (response.choices[0].message.content or "").strip()
    sanitised = _sanitise_llm_json(raw_text)
    try:
        data = _coerce_report_fields(json.loads(sanitised))
    except json.JSONDecodeError:
        logger.error("JSON parse failed (len=%d), storing raw", len(sanitised))
        data = {"summary": f"[PARSE ERROR] {raw_text[:300]}"}
    return raw_text, data


def _fmt(v: float | None, fmt: str = ".2f") -> str:
    """Format a float or return 'N/A' for None."""
    if v is None:
        return "N/A"
    return format(v, fmt)


def process_image(
    image_path: Path,
    model: str,
    client,
    system_prompt: str,
    no_llm: bool,
) -> dict:
    """Run full pipeline on one image; return comparison dict."""
    from src.analysis.composition import analyse as analyse_composition
    from src.analysis.technical import analyse as analyse_technical
    from src.llm.synthesizer import _build_payload, _compute_quality_tier
    from src.models import AnalysisFeature
    from src.utils.loader import extract_exif, load_image

    size_kb = image_path.stat().st_size / 1024
    logger.info("Processing %s  (%.1f KB)", image_path.name, size_kb)

    t_total = time.perf_counter()

    # Load
    pil_image, bgr_array, tensor = load_image(image_path)
    exif = extract_exif(pil_image)

    # L1 — Technical
    t = time.perf_counter()
    tech = analyse_technical(bgr_array, tensor)
    elapsed_l1 = time.perf_counter() - t
    logger.info("  L1 done  %.1fs", elapsed_l1)

    # L2 — Composition
    t = time.perf_counter()
    comp = analyse_composition(bgr_array, pil_image, exif)
    elapsed_l2 = time.perf_counter() - t
    logger.info("  L2 done  %.1fs  scene=%s", elapsed_l2, comp.scene_type)

    # Build payloads
    new_payload = _build_payload(tech, comp, exif, AnalysisFeature.FULL)
    old_payload = _strip_old_payload(new_payload)

    quality_new = _compute_quality_tier(tech)
    quality_old = {k: v for k, v in quality_new.items() if k not in _NEW_TIER_FIELDS}

    comp_dump = {k: v for k, v in comp.model_dump().items() if k != "saliency_map"}

    result: dict = {
        "image": image_path.name,
        "size_kb": round(size_kb, 1),
        "exif": exif.model_dump(exclude_none=True),
        "new_metrics": {
            "musiq": tech.musiq,
            "horizon_tilt_degrees": comp.horizon_tilt_degrees,
            "scene_type": comp.scene_type,
            "color_harmony_type": comp.color_harmony_type,
            "color_harmony_score": comp.color_harmony_score,
        },
        "quality_tier_old": quality_old,
        "quality_tier_new": quality_new,
        "tech_scores": tech.model_dump(),
        "comp_scores": comp_dump,
        "old_report": None,
        "new_report": None,
        "elapsed": {
            "l1": round(elapsed_l1, 2),
            "l2": round(elapsed_l2, 2),
        },
    }

    if no_llm:
        result["elapsed"]["total"] = round(time.perf_counter() - t_total, 2)
        return result

    # Old LLM call (stripped payload)
    logger.info("  Old LLM call…")
    t = time.perf_counter()
    _, old_report = _call_llm(old_payload, system_prompt, model, client)
    elapsed_old_llm = time.perf_counter() - t
    logger.info("  Old LLM done  %.1fs", elapsed_old_llm)
    result["old_report"] = old_report

    # New LLM call (full payload)
    logger.info("  New LLM call…")
    t = time.perf_counter()
    _, new_report = _call_llm(new_payload, system_prompt, model, client)
    elapsed_new_llm = time.perf_counter() - t
    logger.info("  New LLM done  %.1fs", elapsed_new_llm)
    result["new_report"] = new_report

    result["elapsed"]["old_llm"] = round(elapsed_old_llm, 2)
    result["elapsed"]["new_llm"] = round(elapsed_new_llm, 2)
    result["elapsed"]["total"] = round(time.perf_counter() - t_total, 2)

    return result


def _first_sentences(text: str, n: int = 2) -> str:
    """Return first n sentences of text."""
    if not text:
        return "(empty)"
    parts = text.split(". ")
    snippet = ". ".join(parts[:n]).strip()
    if not snippet.endswith("."):
        snippet += "."
    return snippet


def _musiq_label(v: float | None) -> str:
    if v is None:
        return "N/A"
    if v >= 75:
        label = "excellent"
    elif v >= 60:
        label = "good"
    elif v >= 45:
        label = "average"
    else:
        label = "poor"
    return f"{v:.1f} ({label})"


def generate_summary(results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    no_llm = all(r.get("old_report") is None for r in results if "error" not in r)

    lines = [
        "# FrameIQ: Old vs New Pipeline Comparison",
        f"Generated: {now}",
        "",
        "**Old pipeline**: 12 technical + 12 composition fields — no MUSIQ, no scene/horizon/color-harmony context",
        "**New pipeline**: 13 technical + 17 composition fields — full scene, horizon tilt, color harmony context",
        "*(Both calls use the same L1/L2 analysis and the same system prompt)*",
        "",
        "---",
        "",
        "## Metrics Overview",
        "",
        "| Image | Scene | MUSIQ | Horizon° | Color Harmony | Old overall | New overall |",
        "|-------|-------|-------|----------|---------------|-------------|-------------|",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['image'][:30]} | ERROR | — | — | — | — | — |")
            continue
        nm = r.get("new_metrics", {})
        scene = nm.get("scene_type", "—")
        musiq = nm.get("musiq")
        musiq_str = f"{musiq:.1f}" if musiq is not None else "N/A"
        tilt = nm.get("horizon_tilt_degrees")
        tilt_str = f"{tilt:.1f}°" if tilt is not None else "none"
        harmony = nm.get("color_harmony_type", "—")
        hscore = nm.get("color_harmony_score", 0.0)
        old_q = r.get("quality_tier_old", {}).get("overall", "—")
        new_q = r.get("quality_tier_new", {}).get("overall", "—")
        stem = Path(r["image"]).stem[:30]
        lines.append(
            f"| {stem} | {scene} | {musiq_str} | {tilt_str}"
            f" | {harmony} ({hscore:.2f}) | {old_q} | {new_q} |"
        )

    lines += ["", "---", "", "## Per-Image Analysis", ""]

    for r in results:
        if "error" in r:
            lines += [f"### {r['image']} — ERROR", "", f"```\n{r['error']}\n```", "", "---", ""]
            continue

        nm = r.get("new_metrics", {})
        lines.append(f"### {r['image']} ({r.get('size_kb', 0):.0f} KB)")
        lines.append("")

        # EXIF
        exif = r.get("exif", {})
        exif_parts = []
        if exif.get("camera_model"):
            exif_parts.append(exif["camera_model"])
        if exif.get("iso"):
            exif_parts.append(f"ISO {exif['iso']}")
        if exif.get("shutter_speed"):
            exif_parts.append(f"{exif['shutter_speed']}s")
        if exif.get("aperture"):
            exif_parts.append(f"f/{exif['aperture']}")
        if exif_parts:
            lines.append(f"**EXIF**: {', '.join(exif_parts)}")
            lines.append("")

        # ── New metrics ───────────────────────────────────────────────────────
        tilt = nm.get("horizon_tilt_degrees")
        if tilt is not None:
            tilt_desc = f"{tilt:.2f}° ({'CCW' if tilt < 0 else 'CW'} tilt)"
        else:
            tilt_desc = "no clear horizon detected"

        lines += [
            "**New metrics (dev-branch only):**",
            f"- MUSIQ: {_musiq_label(nm.get('musiq'))}",
            f"- Horizon tilt: {tilt_desc}",
            f"- Scene type: {nm.get('scene_type', '—')}",
            f"- Color harmony: {nm.get('color_harmony_type', '—')} (score: {nm.get('color_harmony_score', 0.0):.2f})",
            "",
        ]

        # ── Quality tier comparison ───────────────────────────────────────────
        qt_old = r.get("quality_tier_old", {})
        qt_new = r.get("quality_tier_new", {})
        all_keys = list(qt_old.keys()) + [k for k in qt_new if k not in qt_old]
        lines += [
            "**Quality tier comparison:**",
            "",
            "| Dimension | Old | New |",
            "|-----------|-----|-----|",
        ]
        for k in all_keys:
            old_v = qt_old.get(k, "—")
            new_v = qt_new.get(k, "—")
            changed = " ◄ new" if old_v == "—" else (" ◄ changed" if old_v != new_v else "")
            lines.append(f"| {k} | {old_v} | {new_v}{changed} |")
        lines.append("")

        # ── Technical scores ──────────────────────────────────────────────────
        tech = r.get("tech_scores", {})
        brisque = _fmt(tech.get("brisque"), ".1f")
        nima = _fmt(tech.get("nima_aesthetic"), ".2f")
        clip = _fmt(tech.get("clip_iqa"), ".3f")
        musiq = _musiq_label(tech.get("musiq"))
        sharpness = _fmt(tech.get("sharpness_laplacian"), ".1f")
        noise = _fmt(tech.get("noise_sigma"), ".2f")
        exp_mean = _fmt(tech.get("histogram_mean"), ".1f")
        exp_hl = _fmt(tech.get("exposure_clipped_highlights_pct"), ".1f")
        exp_sh = _fmt(tech.get("exposure_clipped_shadows_pct"), ".1f")
        dr = _fmt(tech.get("dynamic_range_stops"), ".2f")
        contrast = _fmt(tech.get("contrast_rms"), ".3f")

        lines += [
            "**Technical scores (both pipelines share these):**",
            "",
            "| Metric | Value | Notes |",
            "|--------|-------|-------|",
            f"| BRISQUE | {brisque} | 0–100 lower=better; <30 excellent |",
            f"| NIMA | {nima} | 1–10 higher=better |",
            f"| CLIP-IQA | {clip} | 0–1 higher=better |",
            f"| MUSIQ *(new)* | {musiq} | 0–100 higher=better |",
            f"| Sharpness (Laplacian) | {sharpness} | >500=sharp |",
            f"| Noise σ | {noise} | <3=clean |",
            f"| Histogram mean | {exp_mean} | 80–170=well exposed |",
            f"| Highlights clipped % | {exp_hl}% | <2=acceptable |",
            f"| Shadows clipped % | {exp_sh}% | <2=acceptable |",
            f"| Dynamic range (stops) | {dr} | >5=rich |",
            f"| Contrast RMS | {contrast} | 0.2–0.5=normal |",
            "",
        ]

        # ── Composition scores ────────────────────────────────────────────────
        comp = r.get("comp_scores", {})
        lines += [
            "**Composition scores:**",
            "",
            "| Metric | Value | Pipeline |",
            "|--------|-------|----------|",
            f"| Best alignment | {comp.get('best_alignment', '—')} | both |",
            f"| RoT alignment score | {_fmt(comp.get('rot_alignment_score'), '.3f')} | both |",
            f"| Golden ratio score | {_fmt(comp.get('golden_ratio_alignment_score'), '.3f')} | both |",
            f"| Centroid X/Y | {_fmt(comp.get('saliency_centroid_x'), '.3f')} / {_fmt(comp.get('saliency_centroid_y'), '.3f')} | both |",
            f"| Negative space | {_fmt(comp.get('negative_space_ratio'), '.3f')} | both |",
            f"| Visual weight balance | {_fmt(comp.get('visual_weight_balance'), '.2f')} | both |",
            f"| Symmetry H/V | {_fmt(comp.get('symmetry_horizontal'), '.3f')} / {_fmt(comp.get('symmetry_vertical'), '.3f')} | both |",
            f"| Line pattern | {comp.get('line_pattern', '—')} | both |",
            f"| Leading lines → subject | {comp.get('leading_lines_converge_to_subject', '—')} | both |",
            f"| Horizon tilt *(new)* | {_fmt(nm.get('horizon_tilt_degrees'), '.2f')}° | new only |",
            f"| Scene type *(new)* | {nm.get('scene_type', '—')} | new only |",
            f"| Color harmony *(new)* | {nm.get('color_harmony_type', '—')} ({_fmt(nm.get('color_harmony_score'), '.2f')}) | new only |",
            "",
        ]

        # ── LLM comparison ────────────────────────────────────────────────────
        old_report = r.get("old_report")
        new_report = r.get("new_report")

        if no_llm or (old_report is None and new_report is None):
            lines += [
                "*(LLM comparison skipped — run without `--no-llm` to compare critiques)*",
                "",
            ]
        else:
            lines += [
                "**Summary comparison:**",
                "",
                f"*Old:* {_first_sentences(old_report.get('summary', '') if old_report else '', 2)}",
                "",
                f"*New:* {_first_sentences(new_report.get('summary', '') if new_report else '', 2)}",
                "",
            ]

            report_keys = [
                ("composition", "Composition"),
                ("aesthetics", "Aesthetics"),
                ("technical", "Technical"),
                ("improvements", "Improvements"),
                ("editing", "Editing"),
                ("inspiration", "Inspiration"),
            ]
            for key, label in report_keys:
                old_val = (old_report or {}).get(key, "")
                new_val = (new_report or {}).get(key, "")
                if not old_val and not new_val:
                    continue
                lines += [
                    f"<details>",
                    f"<summary><strong>{label}</strong></summary>",
                    "",
                    "**Old pipeline:**",
                    "",
                    f"{old_val or '*(not included)*'}",
                    "",
                    "**New pipeline:**",
                    "",
                    f"{new_val or '*(not included)*'}",
                    "",
                    "</details>",
                    "",
                ]

        elapsed = r.get("elapsed", {})
        parts = [f"L1={elapsed.get('l1', '?')}s", f"L2={elapsed.get('l2', '?')}s"]
        if "old_llm" in elapsed:
            parts.append(f"Old LLM={elapsed['old_llm']}s")
        if "new_llm" in elapsed:
            parts.append(f"New LLM={elapsed['new_llm']}s")
        if "total" in elapsed:
            parts.append(f"**Total={elapsed['total']}s**")
        lines += [f"*Timing: {', '.join(parts)}*", "", "---", ""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old vs new FrameIQ pipeline on a batch of images"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory of images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls — compare CV metrics only",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging (per-metric timing)",
    )
    args = parser.parse_args()

    _setup_logging(args.debug)

    image_paths = sorted(
        p for p in args.dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        logger.error("No images found in %s", args.dir)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_paths), args.dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    if not args.no_llm:
        from src.llm.client import _MODEL, get_client
        from src.llm.synthesizer import _SYSTEM_PROMPT

        client = get_client()
        system_prompt = _SYSTEM_PROMPT
        model = _MODEL
        logger.info("LLM model: %s", model)
    else:
        client = None
        system_prompt = ""
        model = ""

    results: list[dict] = []

    for i, image_path in enumerate(image_paths, 1):
        logger.info("[%d/%d] %s", i, len(image_paths), image_path.name)
        try:
            result = process_image(
                image_path,
                model=model,
                client=client,
                system_prompt=system_prompt,
                no_llm=args.no_llm,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", image_path.name, exc, exc_info=True)
            result = {
                "image": image_path.name,
                "error": str(exc),
                "size_kb": image_path.stat().st_size / 1024,
            }

        results.append(result)

        stem = image_path.stem
        out_file = out_dir / f"{stem}.json"
        out_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("  Saved %s", out_file.name)

    summary = generate_summary(results)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary: %s", summary_path)

    print("\n" + "=" * 72)
    print(summary)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
