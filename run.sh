#!/bin/bash
# ═══════════════════════════════════════════
#  Vibe ML — Local Development
#  Usage: chmod +x run.sh && ./run.sh
# ═══════════════════════════════════════════

set -e

echo "═══ Vibe ML — Local Dev ═══"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install it first."
    exit 1
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install deps
echo "→ Installing dependencies..."
pip install -r requirements.txt -q

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "→ Created .env from .env.example"
fi

# Create dirs
mkdir -p storage/uploads storage/outputs storage/models frontend

# Run
echo ""
echo "═══════════════════════════════════════════"
echo "  Starting Vibe ML..."
echo "  App:  http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════"
echo ""

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
