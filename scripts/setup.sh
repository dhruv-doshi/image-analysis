#!/usr/bin/env bash
# ============================================================
# setup.sh — one-command first-time dev setup
# ============================================================
# Creates the Python venv, installs all deps, copies .env,
# installs frontend dependencies, and registers pre-commit hooks.
#
# Run once after cloning:
#   ./scripts/setup.sh
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Setting up Python virtual environment ==="
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "=== Copying .env ==="
if [ -f .env ]; then
    echo "  .env already exists — skipping copy"
else
    cp .env.example .env
    echo "  .env created from .env.example"
fi

echo "=== Installing frontend dependencies ==="
npm --prefix frontend install --silent

echo "=== Copying frontend/.env ==="
if [ -f frontend/.env ]; then
    echo "  frontend/.env already exists — skipping copy"
else
    cp frontend/.env.example frontend/.env
    echo "  frontend/.env created from frontend/.env.example"
fi

echo "=== Installing pre-commit hooks ==="
if .venv/bin/pre-commit --version &>/dev/null 2>&1; then
    .venv/bin/pre-commit install
else
    echo "  pre-commit not found — skipping hook installation"
    echo "  Install it with: .venv/bin/pip install pre-commit"
fi

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Edit .env and set ANTHROPIC_API_KEY=sk-ant-..."
echo "  2. Edit frontend/.env and set NEXT_PUBLIC_API_URL if needed"
echo "  3. Run the app: ./run_local.sh"
