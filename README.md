# FrameIQ — AI Photo Analysis

An AI-powered Streamlit app that analyses uploaded photographs for composition, technical quality, and aesthetics — then returns a structured natural-language critique with improvement tips and photographer inspiration.

## Features

- **Composition Analysis** — Rule of Thirds / Golden Ratio alignment, leading lines, symmetry, visual weight, negative space (powered by U²-Net saliency via rembg)
- **Technical Assessment** — sharpness, noise, exposure clipping, dynamic range, contrast (BRISQUE, NIMA, CLIP-IQA+)
- **AI Critique** — natural-language report from Claude covering composition, aesthetics, technical quality, editing tips, and photographer inspiration
- **EXIF Display** — camera, lens, ISO, shutter speed, aperture, focal length

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python 3.11+ |
| IQA metrics | pyiqa (BRISQUE, NIMA, CLIP-IQA+) |
| Saliency | rembg (U²-Net, CPU backend) |
| Image processing | Pillow, OpenCV, scikit-image |
| LLM | Anthropic Claude API (`claude-opus-4-6` by default) |
| Data models | Pydantic v2 |

## Project Structure

```
image-analysis/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── pyproject.toml                # ruff, mypy, bandit, pytest config
├── .env.example                  # environment variable template
├── prompts/
│   └── system.md                 # Claude system prompt
├── src/
│   ├── models.py                 # Pydantic models
│   ├── utils/loader.py           # image I/O + EXIF extraction
│   ├── analysis/
│   │   ├── technical.py          # Layer 1: IQA + CV metrics
│   │   └── composition.py        # Layer 2: composition analysis
│   └── llm/
│       ├── client.py             # Anthropic SDK wrapper
│       └── synthesizer.py        # Layer 3: LLM synthesis
└── tests/                        # 130 tests, all mocked — no GPU/API key needed
```

## Setup

### 1. Clone

```bash
git clone https://github.com/dhruv-doshi/image-analysis.git
cd image-analysis
```

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, rembg will automatically download the U²-Net model weights (~170 MB). pyiqa will also download BRISQUE/NIMA/CLIP-IQA+ weights. This only happens once and is cached in `~/.cache/`.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

All other variables are optional (see `.env.example` for details).

### 5. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, upload a JPEG, and wait for the analysis.

## Running Tests

No API key or GPU required — all external dependencies are mocked.

```bash
source .venv/bin/activate
pytest
```

Expected: **130 passed, 7 skipped** (the 7 skipped require the optional `piexif` package).

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `LLM_MODEL` | No | `claude-opus-4-6` | Claude model ID |
| `MAX_IMAGE_SIZE_MB` | No | `10` | Upload size limit |
| `STREAMLIT_SERVER_PORT` | No | `8501` | Streamlit port |
