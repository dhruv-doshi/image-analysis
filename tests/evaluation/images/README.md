# User Images for FrameIQ Evaluation

Drop your own JPEG or PNG photos into **this directory** (`tests/evaluation/images/`), then run:

```bash
python tests/evaluation/run_eval.py --user-images tests/evaluation/images/
```

## Rules

- **Flat directory only** — subfolders are ignored.
- Supported formats: `.jpg`, `.jpeg`, `.png` (case-insensitive).
- No quality labels are needed — the pipeline runs full Layer 1+2+3 analysis and you can
  compare Claude's inferred tier to your own expert judgement.
- Images are never committed (the `images/` directory is gitignored except this README).

## When to add images

You can add images **at any time** — before or after the framework is set up.
The evaluation always generates the five synthetic tier images automatically; your
own photos are added on top of those.

## Typical workflow

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Drop photos into this directory (e.g. my_photo.jpg)

# 3. Run full evaluation (requires ANTHROPIC_API_KEY in .env)
# Layer 1+2 only (fast, no API key)
python tests/evaluation/run_eval.py --no-llm --user-images tests/evaluation/images/ --verbose

# Full pipeline with Claude (needs ANTHROPIC_API_KEY in .env)
python tests/evaluation/run_eval.py --user-images tests/evaluation/images/ --verbose

# 4. Open the report
open tests/evaluation/results/report.md
```

## Output files

After the run, two files are written to `tests/evaluation/results/`:

| File | Description |
|------|-------------|
| `results.json` | Machine-readable data: all metrics, tiers, text analysis, bias stats |
| `report.md` | Human-readable scorecard, bar charts, and prompt suggestions |

## What the evaluation tells you

For each user image the report shows:

- **Numerical tier** — derived from Layer 1+2 scores (BRISQUE, sharpness, noise, exposure)
- **Claude's inferred tier** — extracted from the LLM's text output
- **Tier gap** — how many tiers Claude inflated or deflated relative to the scores
- **Positivity score** — ratio of positive to critical words in Claude's critique
- **Coverage rate** — fraction of poor metrics Claude actually mentioned
