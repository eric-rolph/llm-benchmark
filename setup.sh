#!/usr/bin/env bash
# setup.sh — cross-platform setup script (macOS / Linux)
# Windows: use setup.ps1 instead
set -e

PYTHON=${PYTHON:-python3}

echo "=== LLM Benchmark Suite — Setup ==="

# Check Python version (3.11+)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "ERROR: Python 3.11+ required (found $PY_VER)"
  exit 1
fi
echo "Python $PY_VER — OK"

# Create virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv .venv
else
  echo "Virtual environment already exists — skipping"
fi

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Dry-run:   llm-bench --dry-run"
echo "  Discover:  llm-bench --discover"
echo "  Run all:   llm-bench"
echo ""
