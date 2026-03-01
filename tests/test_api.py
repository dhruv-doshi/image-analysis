"""
Tests for the FastAPI /health and /analyse endpoints.

All pipeline functions are mocked via the api module namespace so no
real model inference or Claude API calls are made.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.models import (
    AnalysisReport,
    CompositionScores,
    ExifData,
    TechnicalScores,
)

# ---------------------------------------------------------------------------
# Minimal fake pipeline outputs
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_exif() -> ExifData:
    return ExifData()


@pytest.fixture()
def fake_tech() -> TechnicalScores:
    return TechnicalScores(
        brisque=40.0,
        nima_aesthetic=6.5,
        clip_iqa=0.7,
        sharpness_laplacian=300.0,
        sharpness_regional={
            "top_left": 300.0,
            "top_right": 300.0,
            "bottom_left": 300.0,
            "bottom_right": 300.0,
        },
        noise_sigma=4.0,
        exposure_clipped_highlights_pct=1.0,
        exposure_clipped_shadows_pct=1.0,
        histogram_mean=128.0,
        histogram_std=50.0,
        dynamic_range_stops=4.0,
        contrast_rms=0.35,
    )


@pytest.fixture()
def fake_comp() -> CompositionScores:
    return CompositionScores(
        saliency_centroid_x=0.5,
        saliency_centroid_y=0.5,
        rot_alignment_score=0.1,
        golden_ratio_alignment_score=0.12,
        best_alignment="rule_of_thirds",
        negative_space_ratio=0.4,
        visual_weight_quadrants={
            "top_left": 0.25,
            "top_right": 0.25,
            "bottom_left": 0.25,
            "bottom_right": 0.25,
        },
        visual_weight_balance=1.0,
        symmetry_horizontal=0.9,
        symmetry_vertical=0.85,
        dominant_line_angles=[45.0],
        leading_lines_converge_to_subject=False,
        line_pattern="diagonal",
    )


@pytest.fixture()
def fake_report() -> AnalysisReport:
    return AnalysisReport(summary="A decent photo with good composition.")


@pytest.fixture()
def fake_quality_tier() -> dict:
    return {
        "overall": "good",
        "brisque_tier": "good",
        "sharpness_tier": "good",
        "noise_tier": "good",
        "exposure_tier": "excellent",
    }


# ---------------------------------------------------------------------------
# Minimal JPEG bytes
# ---------------------------------------------------------------------------

@pytest.fixture()
def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Client + pipeline mocks
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_mocks(fake_exif, fake_tech, fake_comp, fake_report, fake_quality_tier):
    """Patch all six pipeline callables in the api module namespace."""
    fake_pil = MagicMock()
    fake_bgr = MagicMock()
    fake_tensor = MagicMock()

    with (
        patch("api.load_image", return_value=(fake_pil, fake_bgr, fake_tensor)),
        patch("api.extract_exif", return_value=fake_exif),
        patch("api.analyse_technical", return_value=fake_tech),
        patch("api.analyse_composition", return_value=fake_comp),
        patch("api._compute_quality_tier", return_value=fake_quality_tier),
        patch("api.synthesise", return_value=fake_report),
    ):
        yield


@pytest.fixture()
def client():
    from api import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_analyse_valid_jpeg(client, pipeline_mocks, jpeg_bytes):
    resp = client.post(
        "/analyse",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"features": "full"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {"exif", "quality_tier", "technical", "composition", "report"}


def test_analyse_invalid_content_type(client, pipeline_mocks):
    resp = client.post(
        "/analyse",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyse_file_too_large(client, pipeline_mocks):
    big = b"x" * (21 * 1024 * 1024)
    resp = client.post(
        "/analyse",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 400


def test_analyse_missing_file(client):
    resp = client.post("/analyse")
    assert resp.status_code == 422
