# FrameIQ — AI Image Analysis Tool

An AI-powered tool that analyses photographs for composition, aesthetics, and technical quality — then provides actionable improvement tips and contextual inspiration including similar work by renowned photographers.

## Features

- **Composition Analysis** — rule of thirds, leading lines, symmetry, balance, framing
- **Aesthetic Scoring** — colour harmony, contrast, tone, mood
- **Technical Assessment** — exposure, sharpness, depth of field, noise
- **Improvement Tips** — specific, actionable suggestions for each weakness identified
- **Photographer Recommendations** — famous photographers whose style matches the uploaded image
- **Similar Style Examples** — genre, era, and movement context

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python 3.11+ |
| Vision Models | PyTorch / HuggingFace (CLIP, BLIP, etc.) |
| LLM Analysis | Anthropic Claude API |
| Image Processing | Pillow, OpenCV |

## Project Structure

Directories are created incrementally as code is written. Current state:

```
image-analysis/
├── requirements.txt
├── .env.example
├── CLAUDE.md
└── README.md
```

Planned additions as development progresses:

```
├── app.py                  # Streamlit entry point
├── src/
│   ├── analysis/           # Composition, colour, and technical analysers
│   ├── llm/                # LLM prompt builders and API clients
│   ├── models/             # DL model loaders and inference
│   └── utils/              # Shared helpers
├── prompts/                # LLM prompt templates
└── tests/                  # Unit and integration tests
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/dhruv-doshi/image-analysis.git
cd image-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Run the app
streamlit run app.py
```

## Environment Variables

See `.env.example` for all required variables. At minimum you will need an `ANTHROPIC_API_KEY`.
