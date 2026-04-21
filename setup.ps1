$ErrorActionPreference = "Stop"
Write-Host "Setting up LLM Benchmark Suite..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.10+ and add it to PATH."
    exit 1
}

python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv"; exit 1 }

.\.venv\Scripts\Activate.ps1
pip install --upgrade pip -q
pip install -r requirements.txt

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit config.yaml with your model IDs"
Write-Host "  2. Load a model in LM Studio"
Write-Host "  3. .\.venv\Scripts\Activate.ps1"
Write-Host "  4. python run.py --list-models   (verify connection)"
Write-Host "  5. python run.py"
