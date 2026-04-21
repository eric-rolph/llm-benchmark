#!/usr/bin/env bash
# setup.sh — cross-platform setup script (macOS / Linux)
# Windows: use setup.ps1 instead
set -e

PYTHON=${PYTHON:-python3}

echo "=== LLM Benchmark Suite — Setup ==="

# Check Python version (3.10+)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "ERROR: Python 3.10+ required (found $PY_VER)"
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
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Discover:  python run.py --discover"
echo "  Run all:   python run.py"
echo ""
