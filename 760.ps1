$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

# Find Python
$PY_BIN = $null

foreach ($candidate in @("python", "py -3.12", "py -3.11", "py")) {
    try {
        if ($candidate -like "py*") {
            $version = & py --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PY_BIN = "py"
                break
            }
        } else {
            $version = & $candidate --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PY_BIN = $candidate
                break
            }
        }
    } catch {}
}

if (-not $PY_BIN) {
    Write-Host "Python not found. Install Python 3.11 or 3.12 first." -ForegroundColor Red
    exit 1
}

$VENV_DIR = Join-Path $ROOT_DIR ".venv"

$venvPython = Join-Path $VENV_DIR "Scripts\python.exe"

# Create venv if missing
if (!(Test-Path $venvPython)) {
    if (Test-Path $VENV_DIR) {
        Remove-Item -Recurse -Force $VENV_DIR
    }

    if ($PY_BIN -eq "py") {
        & py -m venv $VENV_DIR
    } else {
        & python -m venv $VENV_DIR
    }
}

# Upgrade pip tools
& $venvPython -m pip install --upgrade pip setuptools wheel

# Install dependencies
& $venvPython -m pip install -r (Join-Path $ROOT_DIR "requirements.txt")

Write-Host "`nInstall complete." -ForegroundColor Green
Write-Host "Run RQ1: $venvPython run_rq1.py"
Write-Host "Run RQ2: $venvPython run_rq2.py"
