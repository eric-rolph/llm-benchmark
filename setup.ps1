$ErrorActionPreference = "Stop"
Write-Host "Setting up LLM Benchmark Suite..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.11+ and add it to PATH."
    exit 1
}

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Error "Python 3.11+ required"; exit 1 }

python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv"; exit 1 }

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -q
python -m pip install -e .

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit config.yaml with your model IDs"
Write-Host "  2. Load a model in LM Studio"
Write-Host "  3. .\.venv\Scripts\Activate.ps1"
Write-Host "  4. llm-bench --dry-run           (validate setup)"
Write-Host "  5. llm-bench --discover          (verify backend connection)"
Write-Host "  6. llm-bench"
