#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

export PYTHONUNBUFFERED=1

TEACHER_ROOT="${TEACHER_ROOT:-datasets/sdxl_cloud_teacher_watch46_v4_freeprompt}"
OUT_DIR="${OUT_DIR:-out/tiny_train_watch46_v4_freeprompt_v2_h1536_l64_128}"
STEPS="${STEPS:-16000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LATENT="${LATENT:-64}"
HIDDEN="${HIDDEN:-1536}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
LR="${LR:-0.002}"
DEVICE="${DEVICE:-auto}"
TARGET_DOWNSAMPLE_SIZE="${TARGET_DOWNSAMPLE_SIZE:-96}"
TARGET_BLUR_RADIUS="${TARGET_BLUR_RADIUS:-0.15}"
SMOOTHNESS_LOSS_WEIGHT="${SMOOTHNESS_LOSS_WEIGHT:-0.0002}"
SMOOTHNESS_STEP_PIXELS="${SMOOTHNESS_STEP_PIXELS:-1}"

if [[ ! -f "$TEACHER_ROOT/metadata.jsonl" ]]; then
  echo "missing teacher dataset: $TEACHER_ROOT/metadata.jsonl" >&2
  echo "generate it first with: tools/cloud_generate_sdxl_watch_v4.sh" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

python3 tools/train_tiny_coordinate_mlp.py \
  --teacher-root "$TEACHER_ROOT" \
  --image-size 128 \
  --target-downsample-size "$TARGET_DOWNSAMPLE_SIZE" \
  --target-blur-radius "$TARGET_BLUR_RADIUS" \
  --latent "$LATENT" \
  --hidden "$HIDDEN" \
  --hidden-layers "$HIDDEN_LAYERS" \
  --coord-frequencies 1,2,4,8 \
  --prompt-encoder compositional_v2 \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --smoothness-loss-weight "$SMOOTHNESS_LOSS_WEIGHT" \
  --smoothness-step-pixels "$SMOOTHNESS_STEP_PIXELS" \
  --device "$DEVICE" \
  --progress-every 500 \
  --preview-prompts-file configs/prompt_eval_suite.json \
  --out-dir "$OUT_DIR" \
  --out-json "$OUT_DIR/tiny_weights.json" \
  --out-swift "$OUT_DIR/TinyWeights.swift" \
  --out-bin "$OUT_DIR/TinyWeights.bin"

echo "Training output ready: $OUT_DIR"
