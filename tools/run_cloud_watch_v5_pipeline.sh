#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

RUN_NAME="${RUN_NAME:-sdxl_cloud_teacher_watch46_v5_freeprompt_s3}"
DATASET_DIR="${DATASET_DIR:-datasets/${RUN_NAME}}"
QUALITY_DIR="${QUALITY_DIR:-datasets/${RUN_NAME}_quality_diverse48}"
QUALITY_NO_PROBLEM_DIR="${QUALITY_NO_PROBLEM_DIR:-datasets/${RUN_NAME}_quality_diverse48_no_problem3}"
PROBLEM3_DIR="${PROBLEM3_DIR:-datasets/sdxl_cloud_teacher_watch46_v3_problem3_curated}"
FILTER_MAX_PER_KEY="${FILTER_MAX_PER_KEY:-48}"
TRAIN_STEPS="${TRAIN_STEPS:-28000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8192}"

printf "pipeline_started %s\n" "$(date -Is)"

RUN_NAME="$RUN_NAME" \
VARIANTS_PER_PROMPT="${VARIANTS_PER_PROMPT:-48}" \
SEEDS="${SEEDS:-0,1,2}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash tools/cloud_generate_sdxl_watch_v5.sh

python3 tools/filter_teacher_dataset.py \
  "$DATASET_DIR" \
  "$QUALITY_DIR" \
  --max-per-key "$FILTER_MAX_PER_KEY" \
  --max-per-key-strategy quality_diverse \
  --link-mode hardlink \
  --overwrite

python3 tools/validate_teacher_dataset.py \
  "$QUALITY_DIR" \
  --image-size 128 \
  --max-border-edge-density 0.45 \
  --max-foreground-components 8 \
  --min-largest-foreground-component-ratio 0.45 \
  --allow-invalid

python3 tools/make_teacher_category_contact_sheet.py "$QUALITY_DIR" --image-size 128 --samples-per-key 6

TEACHER_ROOT="$QUALITY_DIR" \
OUT_DIR="out/tiny_train_watch46_v5_quality_diverse48_v2_h1536_l64_128_s${TRAIN_STEPS}" \
DEVICE=cuda \
STEPS="$TRAIN_STEPS" \
BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TARGET_DOWNSAMPLE_SIZE=96 \
TARGET_BLUR_RADIUS=0.15 \
SMOOTHNESS_STEP_PIXELS=1 \
bash tools/train_watch_v4_mlp.sh

python3 tools/filter_teacher_dataset.py \
  "$QUALITY_DIR" \
  "$QUALITY_NO_PROBLEM_DIR" \
  --exclude-keys orange,pizza,bread \
  --link-mode hardlink \
  --overwrite

if [[ -f "$PROBLEM3_DIR/metadata.jsonl" ]]; then
  HYBRID_OUT="out/tiny_train_watch46_v5_quality_diverse48_hybrid_problem3_v2_h1536_l64_128_s${TRAIN_STEPS}"
  mkdir -p "$HYBRID_OUT"
  python3 tools/train_tiny_coordinate_mlp.py \
    --teacher-root "$QUALITY_NO_PROBLEM_DIR" \
    --teacher-root "$PROBLEM3_DIR" \
    --image-size 128 \
    --target-downsample-size 96 \
    --target-blur-radius 0.15 \
    --latent 64 \
    --hidden 1536 \
    --hidden-layers 2 \
    --coord-frequencies 1,2,4,8 \
    --prompt-encoder compositional_v2 \
    --steps "$TRAIN_STEPS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --lr 0.002 \
    --smoothness-loss-weight 0.0002 \
    --smoothness-step-pixels 1 \
    --device cuda \
    --progress-every 500 \
    --preview-prompts-file configs/prompt_eval_suite.json \
    --out-dir "$HYBRID_OUT" \
    --out-json "$HYBRID_OUT/tiny_weights.json" \
    --out-swift "$HYBRID_OUT/TinyWeights.swift" \
    --out-bin "$HYBRID_OUT/TinyWeights.bin"
fi

printf "pipeline_done %s\n" "$(date -Is)"
