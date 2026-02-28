from __future__ import annotations

from enum import Flag, auto

from pydantic import BaseModel


class AnalysisFeature(Flag):
    COMPOSITION = auto()
    AESTHETICS = auto()
    TECHNICAL = auto()
    IMPROVEMENTS = auto()
    EDITING = auto()
    INSPIRATION = auto()
    FULL = COMPOSITION | AESTHETICS | TECHNICAL | IMPROVEMENTS | EDITING | INSPIRATION


class ExifData(BaseModel):
    camera_make: str | None = None
    camera_model: str | None = None
    iso: int | None = None
    shutter_speed: str | None = None  # e.g. "1/125"
    aperture: float | None = None  # e.g. 2.8
    focal_length: float | None = None  # mm
    lens_model: str | None = None
    image_width: int | None = None
    image_height: int | None = None


class TechnicalScores(BaseModel):
    # pyiqa learned metrics
    brisque: float  # 0-100, lower = better
    nima_aesthetic: float | None = None  # 1-10, higher = better
    clip_iqa: float | None = None  # 0-1, higher = better
    # Classical CV
    sharpness_laplacian: float  # variance of Laplacian; higher = sharper
    sharpness_regional: dict  # per-quadrant {top_left, top_right, bottom_left, bottom_right}
    noise_sigma: float  # estimated Gaussian noise std dev
    exposure_clipped_highlights_pct: float  # % pixels near 255
    exposure_clipped_shadows_pct: float  # % pixels near 0
    histogram_mean: float  # mean brightness 0-255
    histogram_std: float  # brightness spread
    dynamic_range_stops: float  # log2(p99/p1)
    contrast_rms: float  # std/mean of grayscale
