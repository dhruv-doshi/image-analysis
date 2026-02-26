# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FrameIQ is a Streamlit web app that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations. Analysis is a two-layer pipeline: local DL vision models (CLIP, BLIP via HuggingFace/PyTorch) feed structured observations into Claude LLM calls that produce the final human-readable report.

## Commands

```bash
# Activate virtual environment (always required first)
source .venv/bin/activate

# Install / sync dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_<module>.py -v

# Run a single test by name
pytest tests/test_<module>.py::test_function_name -v
```

## Architecture

```
app.py                  # Streamlit UI — collects upload, calls pipeline, renders report
src/
  analysis/             # Deterministic image analysers (composition, colour, exposure)
  models/               # DL model loaders/inference wrappers (CLIP, BLIP, etc.)
  llm/                  # Anthropic API client + prompt builders
  utils/                # Image I/O, resizing, shared helpers
prompts/                # Plain-text or Jinja2 prompt templates for LLM calls
tests/                  # Mirrors src/ structure; use pytest
```

**Data flow:** `app.py` → `src/analysis/` (structured metrics) + `src/models/` (embeddings/captions) → `src/llm/` (Claude API, uses `prompts/`) → rendered Streamlit output.

## Key conventions

- All API keys are loaded from `.env` via `python-dotenv`; never hard-code them.
- `uploads/` is gitignored — never commit user images.
- Model weights are downloaded at runtime; the `models/` directory holds configs only.
- The LLM model to use is controlled by the `LLM_MODEL` env var (default: `claude-opus-4-6`).
- Use `src/__init__.py` to expose clean public interfaces from each sub-package.
