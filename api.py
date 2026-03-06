from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any src imports so LLM_MODEL / API keys are already in os.environ
load_dotenv()

# Configure root logger; format is shared by all frameiq/src loggers
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
# basicConfig is a no-op if uvicorn already added root handlers — set levels explicitly
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.getLogger("frameiq").setLevel(_log_level)
logging.getLogger("src").setLevel(_log_level)

# Suppress noisy third-party library loggers
for _noisy in ("httpcore", "httpx", "openai", "python_multipart"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _json_safe(obj: object) -> object:
    """Recursively replace nan/inf floats with None so json.dumps stays valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


logger = logging.getLogger("frameiq.api")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.analysis.composition import analyse as analyse_composition
from src.analysis.technical import analyse as analyse_technical
from src.llm.client import synthesise_stream
from src.llm.synthesizer import _compute_quality_tier, synthesise
from src.models import AnalysisFeature
from src.utils.loader import extract_exif, load_image

_models_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Pre-warm pyiqa and rembg models at startup so the first request isn't slow
    global _models_loaded
    import src.analysis.composition  # noqa: F401 — triggers model pre-warm
    import src.analysis.technical  # noqa: F401 — triggers model pre-warm

    _models_loaded = True
    yield


app = FastAPI(lifespan=lifespan)

# Allow requests from the Next.js dev server; extend via ALLOWED_ORIGINS env var for prod
_origins = ["http://localhost:3000"]
_extra = os.getenv("ALLOWED_ORIGINS", "")
if _extra:
    _origins.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MAX_SIZE = 20 * 1024 * 1024  # 20 MB upload cap
_ALLOWED_TYPES = {"image/jpeg", "image/png"}

# Maps the features= form field values to AnalysisFeature flag bits
_FEATURE_MAP = {
    "composition": AnalysisFeature.COMPOSITION,
    "aesthetics": AnalysisFeature.AESTHETICS,
    "technical": AnalysisFeature.TECHNICAL,
    "improvements": AnalysisFeature.IMPROVEMENTS,
    "editing": AnalysisFeature.EDITING,
    "inspiration": AnalysisFeature.INSPIRATION,
}


def _parse_features(s: str) -> AnalysisFeature:
    # Parse a comma-separated feature list (e.g. "composition,technical") into a flag.
    # "full" or empty string returns FULL (all features enabled).
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts or "full" in parts:
        return AnalysisFeature.FULL
    result = AnalysisFeature(0)
    for p in parts:
        if p in _FEATURE_MAP:
            result |= _FEATURE_MAP[p]
    return result if result else AnalysisFeature.FULL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": _models_loaded}


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    features: str = Form("full"),
) -> dict:
    """Synchronous analysis endpoint.

    Runs the full three-layer pipeline (L1 technical → L2 composition → L3 LLM) and
    returns a single JSON response once everything is complete. Use this when you need
    the full result at once and latency is not a concern.
    """
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"File must be JPEG or PNG, got {file.content_type!r}")
    content = await file.read()
    if len(content) > _MAX_SIZE:
        raise HTTPException(400, "File exceeds 20 MB limit")

    size_kb = len(content) / 1024
    logger.info("POST /analyse  file=%s  size=%.1f KB", file.filename, size_kb)
    t_total = time.perf_counter()

    feature_flag = _parse_features(features)
    suffix = ".jpg" if file.content_type == "image/jpeg" else ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            pil_image, bgr_array, tensor = load_image(tmp_path)
            exif = extract_exif(pil_image)

            t1 = time.perf_counter()
            tech = analyse_technical(bgr_array, tensor)
            logger.info("L1 technical complete  elapsed=%.2fs", time.perf_counter() - t1)

            t2 = time.perf_counter()
            comp = analyse_composition(bgr_array, pil_image, exif)
            logger.info(
                "L2 composition complete  elapsed=%.2fs  scene=%s",
                time.perf_counter() - t2,
                comp.scene_type,
            )

            quality_tier = _compute_quality_tier(tech)
            llm_model = os.getenv("LLM_MODEL", "unknown")
            logger.info("L3 LLM call starting  model=%s", llm_model)
            t3 = time.perf_counter()
            report = synthesise(tech, comp, exif, feature_flag)  # blocking LLM call
            logger.info("L3 LLM call complete  elapsed=%.2fs", time.perf_counter() - t3)
        finally:
            tmp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    logger.info("/analyse complete  total=%.2fs", time.perf_counter() - t_total)
    return {
        "exif": exif.model_dump(),
        "quality_tier": quality_tier,
        "technical": tech.model_dump(),
        "composition": comp.model_dump(),
        "report": report.model_dump(),
    }


@app.post("/analyse/stream")
async def analyse_stream(
    file: UploadFile = File(...),
    features: str = Form("full"),
) -> StreamingResponse:
    """Streaming analysis endpoint (Server-Sent Events).

    Runs L1 (technical) and L2 (composition) synchronously, then immediately pushes
    a 'metrics' SSE event so the frontend can render scores right away. The slow L3
    LLM response is then streamed token-by-token as 'chunk' events, ending with a
    'done' event. This gives much lower perceived latency than /analyse.

    SSE event types emitted:
      {"type": "metrics", "exif": ..., "technical": ..., "composition": ...}
      {"type": "chunk",   "text": "<token>"}   (repeated, one per LLM token)
      {"type": "done"}
      {"type": "error",   "message": "..."}    (only on failure)
    """
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"File must be JPEG or PNG, got {file.content_type!r}")
    content = await file.read()
    if len(content) > _MAX_SIZE:
        raise HTTPException(400, "File exceeds 20 MB limit")

    size_kb = len(content) / 1024
    logger.info("POST /analyse/stream  file=%s  size=%.1f KB", file.filename, size_kb)

    feature_flag = _parse_features(features)
    suffix = ".jpg" if file.content_type == "image/jpeg" else ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            pil_image, bgr_array, tensor = load_image(tmp_path)
            exif = extract_exif(pil_image)

            t1 = time.perf_counter()
            tech = analyse_technical(bgr_array, tensor)
            logger.info("L1 technical complete  elapsed=%.2fs", time.perf_counter() - t1)

            t2 = time.perf_counter()
            comp = analyse_composition(bgr_array, pil_image, exif)
            logger.info(
                "L2 composition complete  elapsed=%.2fs  scene=%s",
                time.perf_counter() - t2,
                comp.scene_type,
            )

            quality_tier = _compute_quality_tier(tech)
        finally:
            tmp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    # Serialise L1+L2 results into the first SSE event sent immediately to the client
    metrics_event = json.dumps(_json_safe({
        "type": "metrics",
        "exif": exif.model_dump(),
        "quality_tier": quality_tier,
        "technical": tech.model_dump(),
        "composition": comp.model_dump(),
    }))

    t_stream_start = time.perf_counter()

    async def generate():
        # Push metrics immediately so the UI can render scores before LLM finishes
        yield f"data: {metrics_event}\n\n"
        logger.info("SSE stream started — metrics event sent")
        chunk_count = 0
        try:
            for chunk in synthesise_stream(tech, comp, exif, feature_flag):
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                chunk_count += 1
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield 'data: {"type":"done"}\n\n'
        logger.info(
            "SSE stream complete  chunks=%d  elapsed=%.2fs",
            chunk_count,
            time.perf_counter() - t_stream_start,
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
