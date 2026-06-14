#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
APPLE_REPO="$ROOT_DIR/.build/ml-stable-diffusion-1.1.1"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
OUTPUT_DIR="$ROOT_DIR/recipes"
CACHE_DIR="$ROOT_DIR/.cache/huggingface"

has_hf_token() {
  [ -n "${HF_TOKEN:-}" ] || \
  [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] || \
  [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] || \
  [ -f "$HOME/.cache/huggingface/token" ] || \
  [ -f "$HOME/.huggingface/token" ]
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run the dependency install step first." >&2
  exit 1
fi

if [ ! -d "$APPLE_REPO/python_coreml_stable_diffusion" ]; then
  echo "Apple repo not found at: $APPLE_REPO" >&2
  exit 1
fi

if ! has_hf_token; then
  echo "No Hugging Face token detected." >&2
  echo "Set HF_TOKEN (or login with huggingface-cli) after accepting the SDXL model license." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_mixed_bit_pre_analysis.py" \
  --apple-repo "$APPLE_REPO" \
  --python-bin "$PYTHON_BIN" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --model-version "stabilityai/stable-diffusion-xl-base-1.0"
