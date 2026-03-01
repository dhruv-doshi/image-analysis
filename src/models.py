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


class CompositionScores(BaseModel):
    saliency_centroid_x: float  # normalised 0-1 from left
    saliency_centroid_y: float  # normalised 0-1 from top
    rot_alignment_score: float  # 0=perfect RoT, 1=worst
    golden_ratio_alignment_score: float  # 0=perfect GR, 1=worst
    best_alignment: str  # "rule_of_thirds" | "golden_ratio"
    negative_space_ratio: float  # 0-1, fraction of non-salient pixels
    visual_weight_quadrants: dict  # {top_left, top_right, bottom_left, bottom_right}
    visual_weight_balance: float  # ratio heaviest/lightest quadrant (1.0=balanced)
    symmetry_horizontal: float  # NCC left vs right halves (1.0=symmetric)
    symmetry_vertical: float  # NCC top vs bottom halves (1.0=symmetric)
    dominant_line_angles: list[float]  # degrees, from HoughLinesP
    leading_lines_converge_to_subject: bool
    line_pattern: str  # "diagonal"|"horizontal"|"vertical"|"mixed"|"none"


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
    brisque_tier: str
    sharpness_tier: str
    noise_tier: str
    exposure_tier: str


class AnalyseResponse(BaseModel):
    exif: ExifData
    quality_tier: QualityTier
    technical: TechnicalScores
    composition: CompositionScores
    report: AnalysisReport
