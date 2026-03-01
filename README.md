# FrameIQ — AI Photo Analysis

AI-powered photo analysis tool. Upload a photograph and receive a structured critique covering composition, technical quality, aesthetics, editing tips, and photographer inspiration — powered by a three-layer pipeline and Claude.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Next.js Frontend (Vercel)                               │
│  – File upload, results display, EXIF viewer             │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTPS / REST
┌─────────────────────▼────────────────────────────────────┐
│  FastAPI Backend (Fly.io)                                │
│                                                          │
│  Layer 1 — Technical (pyiqa + classical CV)              │
│    BRISQUE · NIMA · CLIP-IQA+ · sharpness · noise        │
│    exposure · dynamic range · contrast                   │
│                                                          │
│  Layer 2 — Composition (rembg U²-Net saliency)           │
│    Rule-of-Thirds · Golden Ratio · leading lines         │
│    symmetry · visual weight · negative space             │
│                                                          │
│  Layer 3 — LLM Synthesis (Anthropic Claude)              │
│    structured AnalysisReport JSON from claude-opus-4-6   │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Composition Analysis** — Rule of Thirds / Golden Ratio alignment, leading lines, symmetry, visual weight, negative space (U²-Net saliency via rembg)
- **Technical Assessment** — sharpness, noise, exposure clipping, dynamic range, contrast (BRISQUE, NIMA, CLIP-IQA+)
- **AI Critique** — natural-language report from Claude covering composition, aesthetics, technical quality, editing tips, and photographer inspiration
- **EXIF Display** — camera, lens, ISO, shutter speed, aperture, focal length

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Next.js 16 + React 19 + Tailwind CSS |
| Backend API | FastAPI + uvicorn |
| IQA metrics | pyiqa (BRISQUE, NIMA, CLIP-IQA+) |
| Saliency | rembg (U²-Net, CPU) |
| Image processing | Pillow, OpenCV, scikit-image |
| LLM | Anthropic Claude (`claude-opus-4-6` by default) |
| Data models | Pydantic v2 |
| Backend hosting | Fly.io (Docker) |
| Frontend hosting | Vercel |

## Quick Start (local)

### 1. Clone

```bash
git clone https://github.com/dhruv-doshi/image-analysis.git
cd image-analysis
```

### 2. One-command setup

```bash
./scripts/setup.sh
```

This creates `.venv`, installs Python + frontend dependencies, copies `.env` and `frontend/.env` from their examples, and registers pre-commit hooks.

### 3. Configure environment variables

```bash
# Backend
nano .env                  # set ANTHROPIC_API_KEY=sk-ant-...

# Frontend (only needed if the API runs on a non-default port)
nano frontend/.env         # NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Run

```bash
./run_local.sh
```

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Quality Checks

```bash
./scripts/check_quality.sh
```

Runs in sequence: Ruff lint → Ruff format check → Mypy → Bandit → TypeScript type-check → ESLint → Next.js build check. Exits non-zero on first failure.

## Testing

No API key or GPU required — all external dependencies are mocked.

```bash
source .venv/bin/activate
pytest
```

Expected: **130+ passed, 7 skipped** (the 7 skipped require the optional `piexif` package).

Frontend type-check:

```bash
npm --prefix frontend run type-check
npm --prefix frontend run lint
```

## Deploy to Production

### Prerequisites

| Tool | Install |
|---|---|
| `fly` CLI | https://fly.io/docs/hands-on/install-flyctl/ |
| `vercel` CLI | `npm install -g vercel` |

### Backend → Fly.io

```bash
fly auth login
fly apps create frameiq-api          # first time only
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly secrets set ALLOWED_ORIGINS=https://<your-vercel-url>.vercel.app
fly deploy
```

Health check: `https://frameiq-api.fly.dev/health`

> **Model caching:** `scripts/warmup_models.py` runs at Docker build time and bakes all model weights (pyiqa × 3, rembg U²-Net) into the image at `HF_HOME=/app/.hf_cache`. Cold starts take ~5 s instead of 60 s+. The Docker image is ~2 GB; first `fly deploy` build takes ~5–8 min.

### Frontend → Vercel

```bash
cd frontend
vercel                               # first deploy — follow prompts
```

In the Vercel dashboard → Project Settings → Environment Variables, add:

```
NEXT_PUBLIC_API_URL = https://frameiq-api.fly.dev
```

Subsequent deploys:

```bash
vercel --prod
```

### One-command deploy helper

```bash
./scripts/deploy.sh              # deploy both
./scripts/deploy.sh --backend    # backend only
./scripts/deploy.sh --frontend   # frontend only
```

## Environment Variables

### Backend (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `LLM_MODEL` | No | `claude-opus-4-6` | Claude model ID |
| `MAX_IMAGE_SIZE_MB` | No | `20` | Upload size limit |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS origins |
| `HF_HOME` | No | `~/.cache/huggingface` | Model weight cache directory |
| `STREAMLIT_SERVER_PORT` | No | `8501` | Port when running `app.py` directly |

### Frontend (`frontend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8000` | Backend API base URL |

## Project Structure

```
image-analysis/
├── api.py                        # FastAPI entry point
├── app.py                        # Streamlit entry point (local dev)
├── run_local.sh                  # Start both servers locally
├── requirements.txt
├── pyproject.toml                # ruff, mypy, bandit, pytest config
├── pytest.ini
├── Dockerfile                    # Backend Docker image (Fly.io)
├── fly.toml                      # Fly.io configuration
├── .env.example                  # Backend environment variable template
├── CLAUDE.md
├── README.md
├── prompts/
│   └── system.md                 # Claude system prompt
├── scripts/
│   ├── check_quality.sh          # One-command quality gate
│   ├── setup.sh                  # First-time dev setup
│   ├── deploy.sh                 # Guided deploy helper
│   └── warmup_models.py          # Pre-download model weights (Docker build)
├── src/
│   ├── models.py                 # Pydantic models
│   ├── utils/loader.py           # Image I/O + EXIF extraction
│   ├── analysis/
│   │   ├── technical.py          # Layer 1: IQA + CV metrics
│   │   └── composition.py        # Layer 2: composition analysis
│   └── llm/
│       ├── client.py             # Anthropic SDK wrapper
│       └── synthesizer.py        # Layer 3: LLM synthesis
├── frontend/                     # Next.js 16 app (Vercel)
│   ├── src/app/
│   ├── .env.example
│   └── package.json
└── tests/                        # 130+ tests, all mocked
```
