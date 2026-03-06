from __future__ import annotations

from enum import Flag, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    brisque: float | None = None  # 0-100, lower = better
    nima_aesthetic: float | None = None  # 1-10, higher = better
    clip_iqa: float | None = None  # 0-1, higher = better
    musiq: float | None = None  # 0-100, higher = better
    niqe: float | None = None  # lower = better; <3=excellent, 3-5=good, >8=poor
    # Classical CV
    sharpness_laplacian: float | None = None  # variance of Laplacian; higher = sharper
    sharpness_regional: dict = Field(
        default_factory=dict
    )  # per-quadrant {top_left, top_right, bottom_left, bottom_right}
    noise_sigma: float | None = None  # estimated Gaussian noise std dev
    exposure_clipped_highlights_pct: float | None = None  # % pixels near 255
    exposure_clipped_shadows_pct: float | None = None  # % pixels near 0
    histogram_mean: float | None = None  # mean brightness 0-255
    histogram_std: float | None = None  # brightness spread
    dynamic_range_stops: float | None = None  # log2(p99/p1)
    contrast_rms: float | None = None  # std/mean of grayscale


class CompositionScores(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    saliency_centroid_x: float | None = None  # normalised 0-1 from left
    saliency_centroid_y: float | None = None  # normalised 0-1 from top
    rot_alignment_score: float | None = None  # 0=perfect RoT, 1=worst
    golden_ratio_alignment_score: float | None = None  # 0=perfect GR, 1=worst
    best_alignment: str = "rule_of_thirds"  # "rule_of_thirds" | "golden_ratio"
    negative_space_ratio: float | None = None  # 0-1, fraction of non-salient pixels
    visual_weight_quadrants: dict = Field(
        default_factory=dict
    )  # {top_left, top_right, bottom_left, bottom_right}
    visual_weight_balance: float | None = None  # ratio heaviest/lightest quadrant (1.0=balanced)
    symmetry_horizontal: float | None = None  # NCC left vs right halves (1.0=symmetric)
    symmetry_vertical: float | None = None  # NCC top vs bottom halves (1.0=symmetric)
    dominant_line_angles: list[float] = Field(default_factory=list)  # degrees, from HoughLinesP
    leading_lines_converge_to_subject: bool = False
    line_pattern: str = "none"  # "diagonal"|"horizontal"|"vertical"|"mixed"|"none"
    horizon_tilt_degrees: float | None = None  # signed tilt; 0=level, +/- = clockwise/counter
    scene_type: str = "general"  # portrait|landscape|architecture|macro|general
    dominant_colors: list[list[int]] = Field(default_factory=list)  # top-5 Lab [L,a,b]
    color_harmony_type: str = (
        "complex"  # monochromatic|analogous|complementary|triadic|split-complementary|complex
    )
    color_harmony_score: float = 0.0  # 0–1, higher = stronger match

    # Visualisation only — excluded from model_dump() / JSON serialisation
    saliency_map: Any | None = Field(default=None, exclude=True)


class AnalysisReport(BaseModel):
    summary: str  # always present; 2-3 sentence overall assessment
    composition: str | None = None  # AnalysisFeature.COMPOSITION
    aesthetics: str | None = None  # AnalysisFeature.AESTHETICS
    technical: str | None = None  # AnalysisFeature.TECHNICAL
    improvements: str | None = None  # AnalysisFeature.IMPROVEMENTS
    editing: str | None = None  # AnalysisFeature.EDITING
    inspiration: str | None = None  # AnalysisFeature.INSPIRATION


class QualityTier(BaseModel):
    overall: str
    brisque_tier: str | None = None
    sharpness_tier: str | None = None
    noise_tier: str | None = None
    exposure_tier: str | None = None
    composition_tier: str | None = None


class AnalyseResponse(BaseModel):
    exif: ExifData
    quality_tier: QualityTier
    technical: TechnicalScores
    composition: CompositionScores
    report: AnalysisReport
