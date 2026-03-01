"""
pipeline_runner.py — Run the real FrameIQ pipeline (Layers 1+2+3) on images.

Reuses src/ modules directly. No mocking. One failure per image does not
abort the entire run.

Rate-limit handling
-------------------
When Claude returns HTTP 429 (rate limit), the runner retries up to
LLM_MAX_RETRIES times with exponential backoff starting at LLM_RETRY_BASE_DELAY
seconds. A small inter-image delay (LLM_INTER_IMAGE_DELAY) is inserted
between LLM calls to avoid bursting the API.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models import (
    AnalysisFeature,
    AnalysisReport,
    CompositionScores,
    ExifData,
    TechnicalScores,
)
from src.utils.loader import extract_exif
from tests.evaluation.score_classifier import QualityTier

logger = logging.getLogger(__name__)

# Retry / rate-limit tuning
LLM_MAX_RETRIES = 4
LLM_RETRY_BASE_DELAY = 20   # seconds; doubles each attempt (20, 40, 80, 160)
LLM_INTER_IMAGE_DELAY = 3   # seconds between successive LLM calls


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    image_id: str
    source: str                          # "synthetic" | "user"
    expected_tier: QualityTier | None    # None for user images
    tech: TechnicalScores | None
    comp: CompositionScores | None
    exif: ExifData
    report: AnalysisReport | None        # None if run_llm=False or LLM failed
    pipeline_error: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image)
    return rgb[:, :, ::-1].copy()


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(pil_image).unsqueeze(0)


def _call_llm_with_retry(
    synth_fn,
    image_id: str,
    verbose: bool = False,
) -> AnalysisReport:
    """Call synth_fn() with exponential backoff on HTTP 429 errors.

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return synth_fn()
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < LLM_MAX_RETRIES - 1:
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                msg = (
                    f"  [{image_id}] Rate limited (attempt {attempt + 1}/{LLM_MAX_RETRIES}). "
                    f"Waiting {delay}s..."
                )
                print(msg, flush=True)
                logger.warning(msg)
                time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    images: list[tuple[str, Image.Image, QualityTier | None, str]],
    run_llm: bool = True,
    verbose: bool = False,
) -> list[PipelineResult]:
    """Run the three-layer pipeline on a list of images.

    Args:
        images:  List of (image_id, pil_image, expected_tier_or_None, source).
                 source is "synthetic" or "user".
        run_llm: If False, skip Layer 3 (no API call).
        verbose: If True, log progress to stdout.

    Returns:
        List of PipelineResult, one per image. Errors are captured per image.
    """
    # Lazy imports so module-level pyiqa/rembg init only happens when needed
    from src.analysis import technical as tech_module
    from src.analysis import composition as comp_module
    from src.llm import synthesizer as synth_module

    results: list[PipelineResult] = []
    n = len(images)
    llm_call_count = 0

    for i, (image_id, pil_image, expected_tier, source) in enumerate(images):
        if verbose:
            print(f"[{i + 1}/{n}] Running {image_id}...", flush=True)

        exif = ExifData(image_width=pil_image.width, image_height=pil_image.height)
        tech: TechnicalScores | None = None
        comp: CompositionScores | None = None
        report: AnalysisReport | None = None
        error: str | None = None

        try:
            # EXIF
            try:
                exif = extract_exif(pil_image)
            except Exception as exc:
                logger.warning("[%s] EXIF extraction failed: %s", image_id, exc)

            # Layer 1 — Technical
            bgr = _pil_to_bgr(pil_image)
            tensor = _pil_to_tensor(pil_image)
            tech = tech_module.analyse(bgr, tensor)

            if verbose and tech is not None:
                print(
                    f"  L1: brisque={tech.brisque:.1f}  "
                    f"sharpness={tech.sharpness_laplacian:.0f}  "
                    f"noise={tech.noise_sigma:.2f}  "
                    f"hl={tech.exposure_clipped_highlights_pct:.1f}%  "
                    f"sh={tech.exposure_clipped_shadows_pct:.1f}%",
                    flush=True,
                )

            # Layer 2 — Composition
            comp = comp_module.analyse(bgr, pil_image)

            # Layer 3 — LLM synthesis (with rate-limit retry)
            if run_llm and tech is not None and comp is not None:
                # Small delay between LLM calls to avoid bursting the API
                if llm_call_count > 0:
                    time.sleep(LLM_INTER_IMAGE_DELAY)

                try:
                    report = _call_llm_with_retry(
                        lambda t=tech, c=comp, e=exif: synth_module.synthesise(
                            t, c, e, AnalysisFeature.FULL
                        ),
                        image_id=image_id,
                        verbose=verbose,
                    )
                    llm_call_count += 1
                except Exception as exc:
                    logger.warning("[%s] LLM synthesis failed: %s", image_id, exc)
                    error = f"LLM error: {exc}"
                    llm_call_count += 1  # still counts — we made the attempt

        except Exception as exc:
            logger.error("[%s] Pipeline failed: %s", image_id, exc, exc_info=True)
            error = str(exc)

        results.append(PipelineResult(
            image_id=image_id,
            source=source,
            expected_tier=expected_tier,
            tech=tech,
            comp=comp,
            exif=exif,
            report=report,
            pipeline_error=error,
        ))

    return results
