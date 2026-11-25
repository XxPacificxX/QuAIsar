<#
.SYNOPSIS
    Create a Python virtual environment for QuAIsar and install its dependencies (Windows).

.EXAMPLE
    .\setup.ps1
    Creates .venv inside this folder.

.EXAMPLE
    .\setup.ps1 -VenvPath C:\venvs\quaisar
    Puts the environment at a short path (helps if Windows path-length limits bite).

.EXAMPLE
    .\setup.ps1 -Python py -PythonArgs '-3.12'
    Uses a specific Python through the py launcher.
#>
param(
    [string]$VenvPath = ".venv",
    [string]$Python = "python",
    [string]$PythonArgs = ""
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {

$pyArgs = @()
if ($PythonArgs) { $pyArgs = $PythonArgs -split " " }

$version = & $Python @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]"3.10") {
    throw "Python 3.10+ is required (found $version). Pass -Python to choose another interpreter."
}

Write-Host "Creating virtual environment at $VenvPath (Python $version) ..."
& $Python @pyArgs -m venv $VenvPath
$venvPy = Join-Path $VenvPath "Scripts\python.exe"

& $venvPy -m pip install --upgrade pip --quiet
Write-Host "Installing PyTorch (CPU build) ..."
& $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "Installing the remaining requirements ..."
& $venvPy -m pip install -r requirements.txt --quiet

& $venvPy -c "import torch, pennylane, pymatching, sklearn; print('Check OK: torch', torch.__version__, '| pennylane', pennylane.__version__)"

Write-Host ""
Write-Host "Done. Run a project with:"
Write-Host "  $venvPy p4_qml_ids\run.py"

} finally {
    Pop-Location
}
