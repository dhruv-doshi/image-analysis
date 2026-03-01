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
#   ruff check src/ app.py api.py
#   ruff format --check src/ app.py api.py
#   mypy src/ app.py api.py
#   bandit -c pyproject.toml -r src/ app.py api.py
#   npm --prefix frontend run type-check
#   npm --prefix frontend run lint
#   npm --prefix frontend run build
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Ruff lint ==="
ruff check src/ app.py api.py

echo "=== Ruff format check ==="
ruff format --check src/ app.py api.py

echo "=== Mypy ==="
mypy src/ app.py api.py

echo "=== Bandit ==="
bandit -c pyproject.toml -r src/ app.py api.py

echo ""
echo "=== Frontend: TypeScript type-check ==="
npm --prefix frontend run type-check

echo "=== Frontend: ESLint ==="
npm --prefix frontend run lint

echo "=== Frontend: Build check ==="
npm --prefix frontend run build

echo ""
echo "All checks passed."
