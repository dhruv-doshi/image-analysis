#!/usr/bin/env bash
# ============================================================
# deploy.sh — guided deploy helper for FrameIQ
# ============================================================
# Usage:
#   ./scripts/deploy.sh              # deploy both backend + frontend
#   ./scripts/deploy.sh --backend    # deploy backend to Fly.io only
#   ./scripts/deploy.sh --frontend   # deploy frontend to Vercel only
#
# Prerequisites:
#   Backend:  fly CLI installed, logged in, ANTHROPIC_API_KEY set as secret
#   Frontend: vercel CLI installed, NEXT_PUBLIC_API_URL env var configured
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEPLOY_BACKEND=true
DEPLOY_FRONTEND=true

if [ "${1:-}" = "--backend" ]; then
    DEPLOY_FRONTEND=false
elif [ "${1:-}" = "--frontend" ]; then
    DEPLOY_BACKEND=false
fi

# ── Backend: Fly.io ────────────────────────────────────────────────────────────
deploy_backend() {
    echo "=== Deploying backend to Fly.io ==="

    if ! command -v fly &>/dev/null; then
        echo "ERROR: 'fly' CLI not found."
        echo "Install it from https://fly.io/docs/hands-on/install-flyctl/"
        exit 1
    fi

    if ! fly auth whoami &>/dev/null 2>&1; then
        echo "ERROR: Not logged in to Fly.io. Run: fly auth login"
        exit 1
    fi

    # Verify ANTHROPIC_API_KEY secret exists
    if ! fly secrets list 2>/dev/null | grep -q "ANTHROPIC_API_KEY"; then
        echo "WARNING: ANTHROPIC_API_KEY secret not found in Fly.io."
        echo "Set it with: fly secrets set ANTHROPIC_API_KEY=sk-ant-..."
        read -r -p "Continue anyway? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
    fi

    echo "  Running: fly deploy"
    fly deploy

    echo "  Backend deployed."
    echo "  Health check: https://$(fly info --json 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('Hostname','frameiq-api.fly.dev'))\" 2>/dev/null || echo 'frameiq-api.fly.dev')/health"
}

# ── Frontend: Vercel ───────────────────────────────────────────────────────────
deploy_frontend() {
    echo "=== Deploying frontend to Vercel ==="

    if ! command -v vercel &>/dev/null; then
        echo "ERROR: 'vercel' CLI not found."
        echo "Install it with: npm install -g vercel"
        exit 1
    fi

    if ! vercel whoami &>/dev/null 2>&1; then
        echo "ERROR: Not logged in to Vercel. Run: vercel login"
        exit 1
    fi

    echo "  Running: vercel --prod (from frontend/)"
    cd "$ROOT/frontend"
    vercel --prod

    echo "  Frontend deployed."
    echo "  Remember to set NEXT_PUBLIC_API_URL in Vercel dashboard if not already set."
}

# ── Execute ────────────────────────────────────────────────────────────────────
if $DEPLOY_BACKEND; then
    deploy_backend
fi

if $DEPLOY_FRONTEND; then
    deploy_frontend
fi

echo ""
echo "Deployment complete."
