from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.analysis.composition import analyse as analyse_composition
from src.analysis.technical import analyse as analyse_technical
from src.llm.synthesizer import _compute_quality_tier, synthesise
from src.models import AnalysisFeature
from src.utils.loader import extract_exif, load_image

load_dotenv()

_models_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _models_loaded
    import src.analysis.composition  # noqa: F401 — triggers model pre-warm
    import src.analysis.technical  # noqa: F401 — triggers model pre-warm

    _models_loaded = True
    yield


app = FastAPI(lifespan=lifespan)

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

_MAX_SIZE = 20 * 1024 * 1024
_ALLOWED_TYPES = {"image/jpeg", "image/png"}

_FEATURE_MAP = {
    "composition": AnalysisFeature.COMPOSITION,
    "aesthetics": AnalysisFeature.AESTHETICS,
    "technical": AnalysisFeature.TECHNICAL,
    "improvements": AnalysisFeature.IMPROVEMENTS,
    "editing": AnalysisFeature.EDITING,
    "inspiration": AnalysisFeature.INSPIRATION,
}


def _parse_features(s: str) -> AnalysisFeature:
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
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"File must be JPEG or PNG, got {file.content_type!r}")
    content = await file.read()
    if len(content) > _MAX_SIZE:
        raise HTTPException(400, "File exceeds 20 MB limit")

    feature_flag = _parse_features(features)
    suffix = ".jpg" if file.content_type == "image/jpeg" else ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            pil_image, bgr_array, tensor = load_image(tmp_path)
            exif = extract_exif(pil_image)
            tech = analyse_technical(bgr_array, tensor)
            comp = analyse_composition(bgr_array, pil_image)
            quality_tier = _compute_quality_tier(tech)
            report = synthesise(tech, comp, exif, feature_flag)
        finally:
            tmp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return {
        "exif": exif.model_dump(),
        "quality_tier": quality_tier,
        "technical": tech.model_dump(),
        "composition": comp.model_dump(),
        "report": report.model_dump(),
    }
