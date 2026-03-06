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
    parser.add_argument(
        "--full-output", action="store_true", help="Print complete raw LLM response (no truncation)"
    )
    parser.add_argument(
        "--http", action="store_true",
        help="Also call the live /analyse/stream endpoint and simulate browser SSE parsing",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
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

    quality_tier = _compute_quality_tier(tech, comp)
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
    if args.full_output:
        print(raw)
    else:
        print(raw[:4000])
        if len(raw) > 4000:
            print(f"\n… [{len(raw) - 4000} more chars] — use --full-output to see all")

    # ── Parse report ──────────────────────────────────────────────────────────
    _hr("PARSED REPORT")
    from src.llm.synthesizer import _sanitise_llm_json
    buf = _sanitise_llm_json(raw)

    logger.info("raw chars=%d  sanitised chars=%d", len(raw), len(buf))
    logger.info("sanitised starts with: %s", repr(buf[:80]))
    logger.info("sanitised ends with:   %s", repr(buf[-80:]))

    try:
        report = json.loads(buf)
        print(json.dumps(report, indent=2))
        logger.info("JSON parse: OK  keys=%s", list(report.keys()))
    except json.JSONDecodeError as exc:
        logger.error("JSON parse FAILED: %s", exc)
        logger.error("offending position: %d / %d chars", exc.pos, len(buf))
        ctx_start = max(0, exc.pos - 80)
        ctx_end   = min(len(buf), exc.pos + 80)
        logger.error("context around error:\n%s", repr(buf[ctx_start:ctx_end]))
        logger.error("last 200 chars of sanitised buffer:\n%s", repr(buf[-200:]))
        sys.exit(1)

    # ── SSE round-trip simulation (replicates frontend behaviour) ────────────
    _hr("SSE ROUND-TRIP SIMULATION")
    # Encode every chunk exactly as api.py does, then decode as api.ts does,
    # then apply the JavaScript-equivalent sanitisation, then parse.
    sse_buf = ""
    dropped = 0
    for chunk in chunks:
        frame_json = json.dumps({"type": "chunk", "text": chunk})
        try:
            event = json.loads(frame_json)
        except json.JSONDecodeError:
            dropped += 1
            logger.warning("SSE frame parse FAILED — chunk dropped: %r", frame_json[:120])
            continue
        sse_buf += event["text"]

    logger.info("SSE assembled: %d chars  dropped_frames=%d", len(sse_buf), dropped)
    if sse_buf != raw:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(sse_buf, raw)) if a != b),
            min(len(sse_buf), len(raw)),
        )
        logger.error("SSE round-trip differs from raw!  first diff at char %d", first_diff)
        logger.error("raw[%d:%d]: %r", first_diff, first_diff + 40, raw[first_diff:first_diff + 40])
        logger.error("sse[%d:%d]: %r", first_diff, first_diff + 40, sse_buf[first_diff:first_diff + 40])
    else:
        logger.info("SSE round-trip: identical to raw response ✓")

    # Replicate frontend page.tsx sanitisation exactly
    import re as _re
    fe_buf = sse_buf.strip()
    # Strip code fences
    fe_buf = _re.sub(r'^```(?:json|JSON)?\s*\r?\n?', '', fe_buf)
    fe_buf = _re.sub(r'\r?\n?```[\s\S]*$', '', fe_buf).strip()
    # Extract JSON object (first { to last })
    json_start = fe_buf.find('{')
    json_end   = fe_buf.rfind('}')
    if json_start == -1 or json_end <= json_start:
        logger.error("Frontend simulation: no JSON object found in buffer")
    else:
        fe_buf = fe_buf[json_start:json_end + 1]
        # Fix leading-plus numbers
        fe_buf = _re.sub(r':\s*\+(\d)', r': \1', fe_buf)
        # Add missing commas after ] or } before next "key"
        fe_buf = _re.sub(r'([}\]])\s*\n(\s*"[a-z_]+")', r'\1,\n\2', fe_buf)
        # Remove trailing commas
        fe_buf = _re.sub(r',\s*([}\]])', r'\1', fe_buf)

        logger.info("frontend sanitised: %d chars", len(fe_buf))
        logger.info("frontend buf starts: %s", repr(fe_buf[:80]))
        logger.info("frontend buf ends:   %s", repr(fe_buf[-80:]))

        try:
            fe_report = json.loads(fe_buf)
            logger.info("Frontend simulation: JSON parse OK  keys=%s", list(fe_report.keys()))
        except json.JSONDecodeError as exc:
            logger.error("Frontend simulation: JSON parse FAILED: %s", exc)
            ctx_start = max(0, exc.pos - 80)
            ctx_end   = min(len(fe_buf), exc.pos + 80)
            logger.error("context around error:\n%s", repr(fe_buf[ctx_start:ctx_end]))
            logger.error("last 300 chars:\n%s", repr(fe_buf[-300:]))

    # ── Live HTTP streaming test (--http) ────────────────────────────────────
    if args.http:
        _hr("LIVE HTTP STREAM TEST")
        import re as _re2
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed — run: pip install httpx")
            sys.exit(1)

        url = f"{args.api_url}/analyse/stream"
        logger.info("POST %s", url)
        http_report_buf = ""
        http_chunks = 0
        http_dropped = 0
        http_buf = ""

        with httpx.stream(
            "POST", url,
            files={"file": (args.image.name, args.image.read_bytes(), "image/jpeg")},
            data={"features": "full"},
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                logger.error("HTTP %d: %s", resp.status_code, resp.text)
                sys.exit(1)
            for raw_bytes in resp.iter_bytes():
                http_buf += raw_bytes.decode("utf-8", errors="replace")
                parts = http_buf.split("\n\n")
                http_buf = parts.pop()
                for part in parts:
                    if not part.startswith("data: "):
                        continue
                    try:
                        event = json.loads(part[6:])
                    except json.JSONDecodeError:
                        http_dropped += 1
                        logger.warning("HTTP: SSE frame parse failed — dropped: %r", part[:100])
                        continue
                    if event.get("type") == "chunk":
                        http_report_buf += event["text"]
                        http_chunks += 1
                    elif event.get("type") == "done":
                        logger.info("HTTP: done event received")
                    elif event.get("type") == "error":
                        logger.error("HTTP: server error event: %s", event.get("message"))

        logger.info("HTTP stream: %d chunks  %d chars  %d dropped", http_chunks, len(http_report_buf), http_dropped)

        # Apply frontend-equivalent sanitisation
        fe2 = http_report_buf.strip()
        fe2 = _re2.sub(r'^```(?:json|JSON)?\s*\r?\n?', '', fe2)
        fe2 = _re2.sub(r'\r?\n?```[\s\S]*$', '', fe2).strip()
        js2 = fe2.find('{')
        je2 = fe2.rfind('}')
        if js2 == -1 or je2 <= js2:
            logger.error("HTTP simulation: no JSON object found.  buf[:300]=%r", fe2[:300])
        else:
            fe2 = fe2[js2:je2 + 1]
            fe2 = _re2.sub(r':\s*\+(\d)', r': \1', fe2)
            fe2 = _re2.sub(r'([}\]])\s*\n(\s*"[a-z_]+")', r'\1,\n\2', fe2)
            fe2 = _re2.sub(r',\s*([}\]])', r'\1', fe2)
            logger.info("HTTP sanitised: %d chars", len(fe2))
            logger.info("HTTP buf starts: %s", repr(fe2[:80]))
            logger.info("HTTP buf ends:   %s", repr(fe2[-80:]))
            try:
                http_parsed = json.loads(fe2)
                logger.info("HTTP simulation: JSON parse OK  keys=%s", list(http_parsed.keys()))
            except json.JSONDecodeError as exc:
                logger.error("HTTP simulation: JSON parse FAILED: %s", exc)
                ctx_s = max(0, exc.pos - 80)
                ctx_e = min(len(fe2), exc.pos + 80)
                logger.error("context:\n%s", repr(fe2[ctx_s:ctx_e]))
                logger.error("last 300 chars:\n%s", repr(fe2[-300:]))

    # ── Timing summary ────────────────────────────────────────────────────────
    _hr("TIMING SUMMARY")
    elapsed_total = time.perf_counter() - t_total
    logger.info("L1 technical:   %.2fs", elapsed1)
    logger.info("L2 composition: %.2fs", elapsed2)
    logger.info("L3 LLM:         %.2fs", elapsed3)
    logger.info("total:          %.2fs", elapsed_total)


if __name__ == "__main__":
    main()
