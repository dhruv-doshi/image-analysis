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

# Run the app
streamlit run app.py

```

## Current Repo Structure

Directories are created incrementally as code is written. Only root-level config files exist right now:

```
image-analysis/
├── requirements.txt
├── .env.example
├── CLAUDE.md
└── README.md
```
## Key conventions

- All API keys are loaded from `.env` via `python-dotenv`; never hard-code them.
- `uploads/` is gitignored — never commit user images.
- `temp/` is gitignored — never commit, read, or use any files inside it during processing.
- Model weights are downloaded at runtime; never commit weight files.
- The LLM model to use is controlled by the `LLM_MODEL` env var (default: `claude-opus-4-6`).
