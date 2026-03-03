#!/usr/bin/env python3
"""Compare prompt v1 (original) vs v2 (flaw-focused) on a single image.

Usage:
    python scripts/compare_prompts.py <image_path>
    python scripts/compare_prompts.py          # auto-picks first JPEG in uploads/

Results are saved to: prompts/comparisons/YYYYMMDD_HHMMSS_<image_stem>.json
Add notes from each run to prompts/VERSIONS.md.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import src.llm.synthesizer as _syn
from src.analysis.composition import analyse as analyse_composition
from src.analysis.technical import analyse as analyse_technical
from src.models import AnalysisFeature
from src.utils.loader import extract_exif, load_image

_PROMPT_V1 = (Path(__file__).parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")
_PROMPT_V2 = (Path(__file__).parent.parent / "prompts" / "system_v2.md").read_text(encoding="utf-8")


def _synthesise_with(prompt: str, scores, comp, exif, features):
    """Call synthesise() with an arbitrary system prompt via module-level swap."""
    original = _syn._SYSTEM_PROMPT
    try:
        _syn._SYSTEM_PROMPT = prompt
        return _syn.synthesise(scores, comp, exif, features)
    finally:
        _syn._SYSTEM_PROMPT = original


def _pick_image(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            sys.exit(f"Image not found: {p}")
        return p
    for folder in ("uploads", "temp"):
        folder_path = Path(folder)
        if folder_path.exists():
            candidates = sorted(folder_path.glob("*.jp*g"))
            if candidates:
                print(f"Auto-selected: {candidates[0]}")
                return candidates[0]
    sys.exit(
        "No image path given and no JPEGs found in uploads/.\n"
        "Usage: python scripts/compare_prompts.py <image_path>"
    )


def main() -> None:
    image_path = _pick_image(sys.argv[1] if len(sys.argv) > 1 else None)

    print(f"Image: {image_path}")
    with open(image_path, "rb") as f:
        pil_image, bgr_array, tensor = load_image(io.BytesIO(f.read()))

    print("Running technical analysis…")
    exif = extract_exif(pil_image)
    scores = analyse_technical(bgr_array, tensor)

    print("Running composition analysis…")
    comp = analyse_composition(bgr_array, pil_image)

    features = AnalysisFeature.FULL

    print("Calling LLM — v1 (original prompt)…")
    report_v1 = _synthesise_with(_PROMPT_V1, scores, comp, exif, features)

    print("Calling LLM — v2 (flaw-focused prompt)…")
    report_v2 = _synthesise_with(_PROMPT_V2, scores, comp, exif, features)

    result = {
        "image": str(image_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "v1_original": report_v1.model_dump(exclude_none=True),
        "v2_flaw_focused": report_v2.model_dump(exclude_none=True),
    }

    out_dir = Path("prompts/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{stamp}_{image_path.stem}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nSaved → {out_file}")
    print(f"\n{'─' * 60}")
    print(f"v1 summary:\n  {report_v1.summary}")
    print(f"\nv2 summary:\n  {report_v2.summary}")
    print(f"{'─' * 60}")
    print("\nAdd your observations to prompts/VERSIONS.md.")


if __name__ == "__main__":
    main()
