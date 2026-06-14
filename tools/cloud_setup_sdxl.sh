#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-/tmp/tiny-image-model-venv}"
rm -rf .venv
python3 -m venv --system-site-packages "$VENV_DIR"
ln -s "$VENV_DIR" .venv
source "$VENV_DIR/bin/activate"

python3 -m pip install --disable-pip-version-check --upgrade pip
python3 -m pip install --disable-pip-version-check -r requirements/cloud_sdxl.txt

python3 - <<'PY'
import platform
import torch
import diffusers
import transformers
import accelerate

print("python", platform.python_version())
print("torch", torch.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_capability", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("CUDA is not available. Use a RunPod/Lambda PyTorch GPU template, not a CPU template.")
PY

echo "Cloud SDXL environment is ready:"
echo "  source .venv/bin/activate"
