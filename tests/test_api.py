"""
Tests for the FastAPI /health and /analyse endpoints.

All pipeline functions are mocked via the api module namespace so no
real model inference or Claude API calls are made.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
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
    """Patch all pipeline callables in the api module namespace."""
    fake_pil = MagicMock()
    fake_bgr = MagicMock()
    fake_tensor = MagicMock()

    with (
        patch("api.load_image", return_value=(fake_pil, fake_bgr, fake_tensor)),
        patch("api.extract_exif", return_value=fake_exif),
        patch("api._is_photograph", return_value=(True, "")),
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


# ---------------------------------------------------------------------------
# Async tests — executor offload and pre-screen
# ---------------------------------------------------------------------------

_VALID_REPORT_JSON = json.dumps(
    {
        "summary": "A well-composed photograph.",
        "composition": "Good placement.",
        "aesthetics": "Pleasant colours.",
        "technical": "Sharp and clean.",
        "improvements": "Try different angles.",
        "editing": "Slight contrast boost.",
        "inspiration": "Henri Cartier-Bresson.",
    }
)


def _noisy_jpeg(width: int = 200, height: int = 200) -> bytes:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "JPEG")
    return buf.getvalue()


def _white_jpeg(width: int = 200, height: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def async_mock_layers(fake_tech, fake_comp, fake_report):
    """Patch layers + pre-screen for async endpoint tests."""
    with (
        patch("api.analyse_technical", return_value=fake_tech),
        patch("api.analyse_composition", return_value=fake_comp),
        patch("api.synthesise", return_value=fake_report),
        patch("api.synthesise_stream", return_value=iter([_VALID_REPORT_JSON])),
        patch("api._is_photograph", return_value=(True, "")),
    ):
        yield


@pytest_asyncio.fixture
async def async_client():
    from api import app as _app  # noqa: PLC0415

    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as c:
        yield c


async def test_async_analyse_returns_200(async_client, async_mock_layers):
    resp = await async_client.post(
        "/analyse",
        files={"file": ("photo.jpg", _noisy_jpeg(), "image/jpeg")},
        data={"features": "full"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "report" in body
    assert "technical" in body
    assert "composition" in body


async def test_stream_endpoint_emits_metrics_event(async_client, async_mock_layers):
    events: list[dict] = []
    async with async_client.stream(
        "POST",
        "/analyse/stream",
        files={"file": ("photo.jpg", _noisy_jpeg(), "image/jpeg")},
        data={"features": "full"},
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    event_types = [e.get("type") for e in events]
    assert "metrics" in event_types
    assert "done" in event_types
    assert event_types.index("metrics") < event_types.index("done")


async def test_blank_image_returns_422(async_client):
    """Solid-white JPEG triggers Tier-1 stat check (std=0 < 8) → HTTP 422."""
    import src.analysis.technical as tech_mod

    orig_enabled = tech_mod._PRESCREENING_ENABLED
    orig_prescreener = tech_mod._clip_prescreener
    tech_mod._PRESCREENING_ENABLED = True
    tech_mod._clip_prescreener = False  # skip CLIP; Tier 1 is sufficient
    try:
        resp = await async_client.post(
            "/analyse",
            files={"file": ("white.jpg", _white_jpeg(), "image/jpeg")},
            data={"features": "full"},
        )
        assert resp.status_code == 422
    finally:
        tech_mod._PRESCREENING_ENABLED = orig_enabled
        tech_mod._clip_prescreener = orig_prescreener
