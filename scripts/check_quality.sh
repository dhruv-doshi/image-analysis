#!/usr/bin/env bash
# ============================================================
# check_quality.sh — one-command local quality gate
# ============================================================
# Runs all code-quality checks in sequence. A non-zero exit
# from any step aborts the remaining steps (set -e).
#
# Setup (dev only — these are NOT in requirements.txt):
#   pip install ruff mypy bandit pre-commit types-Pillow
#
# Activate pre-commit hooks (run once per clone):
#   pre-commit install
#
# Run all checks:
#   ./scripts/check_quality.sh
#
# Or run individually:
#   ruff check src/ app.py
#   ruff format --check src/ app.py
#   mypy src/ app.py
#   bandit -c pyproject.toml -r src/ app.py
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Ruff lint ==="
ruff check src/ app.py

echo "=== Ruff format check ==="
ruff format --check src/ app.py

echo "=== Mypy ==="
mypy src/ app.py

echo "=== Bandit ==="
bandit -c pyproject.toml -r src/ app.py

echo ""
echo "All checks passed."
