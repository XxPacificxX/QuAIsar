#!/usr/bin/env bash
# Create a Python virtual environment for QuAIsar and install its dependencies (macOS / Linux).
#
#   ./setup.sh                 # creates .venv inside this folder
#   ./setup.sh ~/venvs/quaisar # custom location
#   PYTHON=python3.12 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV="${1:-.venv}"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'

echo "Creating virtual environment at $VENV ..."
"$PY" -m venv "$VENV"
VPY="$VENV/bin/python"

"$VPY" -m pip install --upgrade pip --quiet
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing PyTorch ..."
    "$VPY" -m pip install torch --quiet
else
    echo "Installing PyTorch (CPU build, avoids a multi-GB CUDA download) ..."
    "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi
echo "Installing the remaining requirements ..."
"$VPY" -m pip install -r requirements.txt --quiet

"$VPY" -c "import torch, pennylane, pymatching, sklearn; print('Check OK: torch', torch.__version__, '| pennylane', pennylane.__version__)"

echo
echo "Done. Run a project with:"
echo "  $VPY p4_qml_ids/run.py"
