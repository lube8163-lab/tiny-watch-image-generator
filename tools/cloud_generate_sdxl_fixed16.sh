#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-sdxl_cloud_teacher_fixed16_v1}"
LIMIT="${LIMIT:-16}"
VARIANTS_PER_PROMPT="${VARIANTS_PER_PROMPT:-8}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7}"
STEPS="${STEPS:-20}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-768}"
TARGET_SIZES="${TARGET_SIZES:-256,128,64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
OUT_DIR="${OUT_DIR:-datasets/${RUN_NAME}}"

python3 tools/generate_sdxl_teacher_dataset.py \
  --allow-downloads \
  --limit "$LIMIT" \
  --variants-per-prompt "$VARIANTS_PER_PROMPT" \
  --seeds "$SEEDS" \
  --steps "$STEPS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --target-sizes "$TARGET_SIZES" \
  --batch-size "$BATCH_SIZE" \
  --out-dir "$OUT_DIR" \
  --save-source

python3 tools/validate_teacher_dataset.py "$OUT_DIR" --image-size 128

python3 tools/make_teacher_category_contact_sheet.py "$OUT_DIR" --image-size 128 --samples-per-key 4

echo "Dataset ready: $OUT_DIR"
