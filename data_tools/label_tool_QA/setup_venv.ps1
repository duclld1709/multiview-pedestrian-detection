param([switch]$Cuda)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & python -m venv (Join-Path $here ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create virtual environment. Install Python 3.11 and enable Add Python to PATH." }
}
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip" }
$torchIndex = "https://download.pytorch.org/whl/cpu"
if ($Cuda) { $torchIndex = "https://download.pytorch.org/whl/cu128" }
& $python -m pip install torch torchvision --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $python -m pip install -r (Join-Path $here "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
Write-Host "Setup complete. Run RUN_QA.bat."
