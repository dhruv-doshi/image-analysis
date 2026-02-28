# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FrameIQ is a Streamlit web app that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations.

## Commands

```bash
# Activate virtual environment (always required first)
source .venv/bin/activate

# Install / sync dependencies
pip install -r requirements.txt

# Copy and fill in environment variables (first-time only)
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# Run the app
streamlit run app.py

# Run tests
pytest

# Run a single test file
pytest tests/test_<module>.py -v

# Lint
ruff check src/ app.py

# Format
ruff format src/ app.py
```

## Repo Structure

```
image-analysis/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── pyproject.toml                # ruff, mypy, bandit, pytest config
├── pytest.ini                    # testpaths + pythonpath
├── .env.example                  # env var template
├── CLAUDE.md
├── README.md
├── prompts/
│   └── system.md                 # Claude system prompt (Layer 3)
├── src/
│   ├── models.py                 # Pydantic models: ExifData, TechnicalScores,
│   │                             #   CompositionScores, AnalysisReport, AnalysisFeature
│   ├── utils/
│   │   └── loader.py             # load_image (PIL/BGR/tensor) + extract_exif
│   ├── analysis/
│   │   ├── technical.py          # Layer 1: pyiqa + classical CV metrics
│   │   └── composition.py        # Layer 2: rembg saliency, RoT/GR, lines, symmetry
│   └── llm/
│       ├── __init__.py
│       ├── client.py             # Anthropic SDK wrapper
│       └── synthesizer.py        # Layer 3: payload builder + Claude call
└── tests/
    ├── conftest.py               # fixtures + pyiqa/rembg mocks
    ├── test_models.py
    ├── test_loader.py
    ├── test_technical.py
    ├── test_composition.py
    └── test_llm.py               # all mocked — no API key needed
```

## Architecture

Three-layer pipeline:

1. **Layer 1 — Technical** (`src/analysis/technical.py`): pyiqa learned metrics (BRISQUE, NIMA, CLIP-IQA+) + classical CV (sharpness, noise, exposure, dynamic range, contrast).
2. **Layer 2 — Composition** (`src/analysis/composition.py`): rembg U²-Net saliency → centroid, Rule-of-Thirds / Golden Ratio alignment, negative space, visual weight, symmetry NCC, leading lines Hough.
3. **Layer 3 — LLM Synthesis** (`src/llm/synthesizer.py`): serialises all scores into an annotated JSON payload, calls Claude, parses the structured `AnalysisReport`.

## Key conventions

- All API keys are loaded from `.env` via `python-dotenv`; never hard-code them.
- `uploads/` and `temp/` are gitignored — never commit user images or temp files.
- Model weights (pyiqa, rembg U²-Net) are downloaded at runtime; never commit weight files.
- The LLM model is controlled by the `LLM_MODEL` env var (default: `claude-opus-4-6`).
- pyiqa models are initialised once at module import (module-level `_brisque`, `_nima`, `_clip_iqa`); each call is wrapped in try/except.
- rembg uses a lazy import inside `_saliency_map()`; falls back to a uniform map on failure.
- `prompts/system.md` is read at synthesizer module import time — keep it on disk.
