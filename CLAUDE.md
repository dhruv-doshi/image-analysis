# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FrameIQ is a FastAPI + Next.js web app that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations.

## Commands

```bash
# Activate virtual environment (always required first)
source .venv/bin/activate

# Install / sync dependencies
pip install -r requirements.txt

# Copy and fill in environment variables (first-time only)
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY

# Run the API server
uvicorn api:app --reload
# or use the convenience script:
./run_local.sh

# Run tests
pytest

# Run a single test file
pytest tests/test_<module>.py -v

# Lint
ruff check src/ api.py

# Format
ruff format src/ api.py

# Run LLM strategy comparison on an image (standalone, no server needed)
python scripts/compare_strategies.py path/to/photo.jpg
python scripts/compare_strategies.py path/to/photo.jpg --no-llm   # L1+L2 only
```

## Repo Structure

```
image-analysis/
├── api.py                        # FastAPI entry point
├── frontend/                     # Next.js frontend
├── requirements.txt
├── pyproject.toml                # ruff, mypy, bandit, pytest config
├── pytest.ini                    # testpaths + pythonpath
├── run_local.sh                  # convenience dev-server launcher
├── .env.example                  # env var template
├── CLAUDE.md
├── README.md
├── docs/
│   ├── architecture.md           # Full pipeline architecture reference
│   ├── api-contract.md           # API contract + TypeScript types (v2)
│   ├── improvements.md           # Improvement proposals with status
│   └── plan.md                   # Implementation plan + completion status
├── prompts/
│   ├── system.md                 # LLM system prompt (Layer 3) — critical, image-first
│   ├── compare_system.md         # Lean prompt used in strategy comparison experiment
│   └── comparisons/              # Per-image JSON results from compare_strategies.py
├── scripts/
│   ├── compare_strategies.py     # Standalone: run 3 LLM input strategies on an image
│   ├── make_test_fixtures.py     # Generate tests/fixtures/photo.jpg + blank.jpg
│   └── warmup_models.py          # Pre-warm pyiqa + rembg models
├── src/
│   ├── models.py                 # Pydantic models: ExifData, TechnicalScores,
│   │                             #   CompositionScores, AnalysisReport, AnalysisFeature
│   ├── config/
│   │   └── metrics.py            # Per-metric enable/disable + quality-tier weights
│   ├── utils/
│   │   └── loader.py             # load_image (PIL/BGR/tensor) + extract_exif
│   ├── analysis/
│   │   ├── technical.py          # Layer 1: pyiqa (parallel) + classical CV + pre-screen gate
│   │   └── composition.py        # Layer 2: OpenCV spectral saliency (or rembg), RoT/GR,
│   │                             #   leading lines, horizon tilt, scene type, color harmony
│   └── llm/
│       ├── __init__.py
│       ├── client.py             # OpenRouter client (OpenAI SDK) + synthesise_stream
│       ├── synthesizer.py        # Layer 3: payload builder + multimodal Claude call
│       └── comparison.py         # 3-strategy comparison (metrics_only / +photo / photo_only)
└── tests/
    ├── conftest.py               # fixtures + pyiqa/rembg mocks
    ├── test_models.py
    ├── test_loader.py
    ├── test_technical.py
    ├── test_composition.py
    ├── test_llm.py               # all mocked — no API key needed
    ├── test_api.py               # async endpoint tests (httpx AsyncClient)
    ├── test_config.py
    ├── fixtures/                 # photo.jpg + blank.jpg (generated)
    └── evaluation/
        ├── images/               # 7 real images used for strategy comparison
        └── results/              # Markdown reports from experiments
```

## Architecture

**Entry point**: `api.py` (FastAPI) receives uploaded images and orchestrates the three-layer pipeline; results are served to the Next.js frontend (`frontend/`). All CPU-bound work is offloaded to `asyncio.get_running_loop().run_in_executor()` — the event loop is never blocked.

Three-layer pipeline:

1. **Layer 1 — Technical** (`src/analysis/technical.py`): Pre-screening gate (`_is_photograph`) rejects blank/non-photo images before any expensive computation. Five pyiqa models (BRISQUE, NIMA, CLIP-IQA+, MUSIQ, NIQE) run **in parallel** via `ThreadPoolExecutor` + classical CV (sharpness, noise, exposure, dynamic range, contrast).
2. **Layer 2 — Composition** (`src/analysis/composition.py`): OpenCV spectral-residual saliency (default, ~100 ms) or rembg U²-Net (opt-in via `use_heavy_saliency` config) → centroid, Rule-of-Thirds / Golden Ratio alignment, negative space, visual weight, symmetry NCC, leading lines Hough, horizon tilt, scene classification, color harmony.
3. **Layer 3 — LLM Synthesis** (`src/llm/synthesizer.py`): Serialises all scores into an annotated JSON payload **and encodes the photograph as base64 JPEG** — both are sent together to Claude as a multimodal message. The LLM treats the image as ground truth and the metrics as supporting evidence. Returns a structured `AnalysisReport`.

## Key conventions

- All API keys are loaded from `.env` via `python-dotenv`; never hard-code them.
- LLM calls go through **OpenRouter** using the OpenAI SDK (`OPENROUTER_API_KEY`). The model is set by `LLM_MODEL` env var (default: `anthropic/claude-haiku-4-5-20251001`).
- `uploads/` is gitignored — never commit user images.
- Model weights (pyiqa, rembg U²-Net) are downloaded at runtime; never commit weight files.
- pyiqa models are initialised once at module import (module-level `_brisque`, `_nima`, `_clip_iqa`, `_musiq`, `_niqe`); each call is wrapped in try/except.
- Default saliency uses OpenCV spectral residual — rembg U²-Net is opt-in via `use_heavy_saliency` in `src/config/metrics.py`.
- `prompts/system.md` is read at synthesizer module import time — keep it on disk. It instructs the LLM to be critical, lead with faults, and treat the photograph as ground truth over the metrics.
- `_compute_quality_tier()` applies three hard gates after the weighted average: terrible sharpness → at least "poor"; terrible exposure → at least "poor"; NIMA average or below → at most "good".
- The `saliency_map` field on `CompositionScores` is `exclude=True` — never serialised in API responses.
