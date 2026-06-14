#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-sdxl_cloud_teacher_watch46_v4_freeprompt}"
PROMPTS="${PROMPTS:-configs/sdxl_tiny_teacher_prompts_v4_watch_freeprompt.json}"
LIMIT="${LIMIT:-0}"
VARIANTS_PER_PROMPT="${VARIANTS_PER_PROMPT:-32}"
SEEDS="${SEEDS:-0,1,2,5}"
STEPS="${STEPS:-20}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-768}"
TARGET_SIZES="${TARGET_SIZES:-256,128,64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
MAX_BORDER_STD="${MAX_BORDER_STD:-0}"
MAX_BORDER_EDGE_DENSITY="${MAX_BORDER_EDGE_DENSITY:-0.40}"
MAX_FOREGROUND_COMPONENTS="${MAX_FOREGROUND_COMPONENTS:-8}"
MIN_LARGEST_FOREGROUND_COMPONENT_RATIO="${MIN_LARGEST_FOREGROUND_COMPONENT_RATIO:-0.45}"
OUT_DIR="${OUT_DIR:-datasets/${RUN_NAME}}"

python3 tools/build_watch_v4_prompt_config.py --out "$PROMPTS"

python3 tools/generate_sdxl_teacher_dataset.py \
  --allow-downloads \
  --presets "$PROMPTS" \
  --limit "$LIMIT" \
  --variants-per-prompt "$VARIANTS_PER_PROMPT" \
  --seeds "$SEEDS" \
  --steps "$STEPS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --target-sizes "$TARGET_SIZES" \
  --batch-size "$BATCH_SIZE" \
  --max-border-std "$MAX_BORDER_STD" \
  --max-border-edge-density "$MAX_BORDER_EDGE_DENSITY" \
  --max-foreground-components "$MAX_FOREGROUND_COMPONENTS" \
  --min-largest-foreground-component-ratio "$MIN_LARGEST_FOREGROUND_COMPONENT_RATIO" \
  --no-abort-on-invalid-image \
  --out-dir "$OUT_DIR" \
  --save-source

python3 tools/validate_teacher_dataset.py \
  "$OUT_DIR" \
  --image-size 128 \
  --max-border-std "$MAX_BORDER_STD" \
  --max-border-edge-density "$MAX_BORDER_EDGE_DENSITY" \
  --max-foreground-components "$MAX_FOREGROUND_COMPONENTS" \
  --min-largest-foreground-component-ratio "$MIN_LARGEST_FOREGROUND_COMPONENT_RATIO" \
  --allow-invalid

python3 tools/make_teacher_category_contact_sheet.py "$OUT_DIR" --image-size 128 --samples-per-key 6

echo "Dataset ready: $OUT_DIR"
