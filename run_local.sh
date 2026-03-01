#!/usr/bin/env bash
# run_local.sh — start the FrameIQ backend and frontend locally
# Usage: ./run_local.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$REPO_DIR/frontend"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
VENV_UVICORN="$REPO_DIR/.venv/bin/uvicorn"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[frameiq]${NC} $*"; }
warn()  { echo -e "${YELLOW}[frameiq]${NC} $*"; }
error() { echo -e "${RED}[frameiq]${NC} $*"; exit 1; }

# ── Cleanup on exit ───────────────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  echo ""
  info "Shutting down..."
  [[ -n "$BACKEND_PID" ]]  && kill "$BACKEND_PID"  2>/dev/null && info "Backend stopped."
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null && info "Frontend stopped."
}
trap cleanup EXIT INT TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[[ -f "$VENV_PYTHON" ]]   || error "Python venv not found at .venv/. Run: python3 -m venv .venv && pip install -r requirements.txt"
[[ -f "$VENV_UVICORN" ]]  || error "uvicorn not found in .venv/. Run: pip install -r requirements.txt"
command -v node &>/dev/null || error "node is not installed. Install Node.js 18+."
command -v npm  &>/dev/null || error "npm is not installed."

if [[ ! -f "$REPO_DIR/.env" ]]; then
  warn ".env not found — copying from .env.example. Fill in ANTHROPIC_API_KEY before using the app."
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  info "Installing frontend dependencies (first run)..."
  npm --prefix "$FRONTEND_DIR" install --silent
fi

# ── Start backend ─────────────────────────────────────────────────────────────
info "Starting backend on http://localhost:8000 ..."
"$VENV_UVICORN" api:app --host 0.0.0.0 --port 8000 \
  --app-dir "$REPO_DIR" \
  --log-level warning &
BACKEND_PID=$!

# Wait until /health responds
for i in {1..15}; do
  sleep 1
  if curl -sf http://localhost:8000/health -o /dev/null 2>/dev/null; then
    info "Backend ready.  GET http://localhost:8000/health"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    error "Backend crashed on startup. Check the output above."
  fi
  [[ $i -eq 15 ]] && error "Backend did not become healthy after 15 s."
done

# ── Start frontend ────────────────────────────────────────────────────────────
info "Starting frontend on http://localhost:3000 ..."
npm --prefix "$FRONTEND_DIR" run dev &
FRONTEND_PID=$!

# Wait until port 3000 responds
for i in {1..20}; do
  sleep 1
  if curl -sf http://localhost:3000 -o /dev/null 2>/dev/null; then
    info "Frontend ready. Open http://localhost:3000"
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    error "Frontend crashed on startup. Check the output above."
  fi
  [[ $i -eq 20 ]] && error "Frontend did not become ready after 20 s."
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}Backend  →${NC}  http://localhost:8000"
echo -e "  ${GREEN}Frontend →${NC}  http://localhost:3000"
echo ""
info "Press Ctrl+C to stop both servers."

# Keep script alive so trap fires on Ctrl+C
wait
