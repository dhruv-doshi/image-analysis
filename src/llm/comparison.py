from __future__ import annotations

import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.llm.client import _MODEL, get_client
from src.llm.synthesizer import _build_payload
from src.models import AnalysisFeature, CompositionScores, ExifData, TechnicalScores

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_COMPARE_SYSTEM_PROMPT: str = (_PROMPTS_DIR / "compare_system.md").read_text(encoding="utf-8")


def _call(messages: list[dict]) -> dict:
    """Call the LLM and return the parsed JSON response dict."""
    client = get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": messages},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.debug("comparison LLM raw:\n%s", raw)

    # Strip markdown fences and extract JSON object
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]

    return json.loads(raw)


def _pil_to_b64_jpeg(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def synthesise_metrics_only(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
) -> dict:
    payload = _build_payload(tech, comp, exif, AnalysisFeature.FULL)
    logger.info("comparison: metrics_only  payload=%d chars", len(payload))
    return _call([{"type": "text", "text": payload}])


def synthesise_metrics_and_photo(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
    pil_image: Image.Image,
) -> dict:
    payload = _build_payload(tech, comp, exif, AnalysisFeature.FULL)
    b64 = _pil_to_b64_jpeg(pil_image)
    logger.info(
        "comparison: metrics_and_photo  payload=%d chars  image_b64=%d chars",
        len(payload),
        len(b64),
    )
    return _call([
        {"type": "text", "text": payload},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])


def synthesise_photo_only(pil_image: Image.Image) -> dict:
    b64 = _pil_to_b64_jpeg(pil_image)
    logger.info("comparison: photo_only  image_b64=%d chars", len(b64))
    return _call([
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])


def run_comparison(
    tech: TechnicalScores,
    comp: CompositionScores,
    exif: ExifData,
    pil_image: Image.Image,
) -> dict:
    """Run all three LLM strategies in parallel and return results side-by-side."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_metrics = executor.submit(synthesise_metrics_only, tech, comp, exif)
        f_both = executor.submit(synthesise_metrics_and_photo, tech, comp, exif, pil_image)
        f_photo = executor.submit(synthesise_photo_only, pil_image)

    return {
        "metrics_only": f_metrics.result(),
        "metrics_and_photo": f_both.result(),
        "photo_only": f_photo.result(),
    }
