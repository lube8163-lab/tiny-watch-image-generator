#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

export PYTHONUNBUFFERED=1

BASE_RUN_NAME="${BASE_RUN_NAME:-sdxl_cloud_teacher_watch75_v6_freeprompt_v7base_s1}"
BASE_DATASET_DIR="${BASE_DATASET_DIR:-datasets/${BASE_RUN_NAME}}"
BASE_QUALITY_DIR="${BASE_QUALITY_DIR:-datasets/${BASE_RUN_NAME}_quality_diverse40}"
BASE_QUALITY_NO_PROBLEM_DIR="${BASE_QUALITY_NO_PROBLEM_DIR:-datasets/${BASE_RUN_NAME}_quality_diverse40_no_problem3}"

FOCUS_RUN_NAME="${FOCUS_RUN_NAME:-sdxl_cloud_teacher_watch75_v7_focus_s1}"
FOCUS_DATASET_DIR="${FOCUS_DATASET_DIR:-datasets/${FOCUS_RUN_NAME}}"
FOCUS_QUALITY_DIR="${FOCUS_QUALITY_DIR:-datasets/${FOCUS_RUN_NAME}_quality_diverse24}"

PROBLEM3_DIR="${PROBLEM3_DIR:-datasets/sdxl_cloud_teacher_watch46_v3_problem3_curated}"
TRAIN_STEPS="${TRAIN_STEPS:-36000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8192}"
HIDDEN="${HIDDEN:-1536}"
FOCUS_FILTER_MAX_PER_KEY="${FOCUS_FILTER_MAX_PER_KEY:-24}"
BASE_FILTER_MAX_PER_KEY="${BASE_FILTER_MAX_PER_KEY:-40}"

printf "v7_pipeline_started %s\n" "$(date -Is)"

RUN_NAME="$BASE_RUN_NAME" \
OUT_DIR="$BASE_DATASET_DIR" \
VARIANTS_PER_PROMPT="${BASE_VARIANTS_PER_PROMPT:-40}" \
SEEDS="${BASE_SEEDS:-0,1}" \
STEPS="${BASE_STEPS:-20}" \
TARGET_SIZES="${TARGET_SIZES:-128}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash tools/cloud_generate_sdxl_watch_v6.sh

python3 tools/filter_teacher_dataset.py \
  "$BASE_DATASET_DIR" \
  "$BASE_QUALITY_DIR" \
  --max-per-key "$BASE_FILTER_MAX_PER_KEY" \
  --max-per-key-strategy quality_diverse \
  --link-mode hardlink \
  --overwrite

python3 tools/validate_teacher_dataset.py \
  "$BASE_QUALITY_DIR" \
  --image-size 128 \
  --max-border-edge-density 0.45 \
  --max-foreground-components 8 \
  --min-largest-foreground-component-ratio 0.45 \
  --allow-invalid

python3 tools/make_teacher_category_contact_sheet.py "$BASE_QUALITY_DIR" --image-size 128 --samples-per-key 6

python3 tools/filter_teacher_dataset.py \
  "$BASE_QUALITY_DIR" \
  "$BASE_QUALITY_NO_PROBLEM_DIR" \
  --exclude-keys orange,pizza,bread \
  --link-mode hardlink \
  --overwrite

RUN_NAME="$FOCUS_RUN_NAME" \
OUT_DIR="$FOCUS_DATASET_DIR" \
VARIANTS_PER_PROMPT="${FOCUS_VARIANTS_PER_PROMPT:-32}" \
SEEDS="${FOCUS_SEEDS:-10,11}" \
STEPS="${FOCUS_STEPS:-22}" \
TARGET_SIZES="${TARGET_SIZES:-128}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash tools/cloud_generate_sdxl_watch_v7_focus.sh

python3 tools/filter_teacher_dataset.py \
  "$FOCUS_DATASET_DIR" \
  "$FOCUS_QUALITY_DIR" \
  --max-per-key "$FOCUS_FILTER_MAX_PER_KEY" \
  --max-per-key-strategy quality_diverse \
  --link-mode hardlink \
  --overwrite

python3 tools/validate_teacher_dataset.py \
  "$FOCUS_QUALITY_DIR" \
  --image-size 128 \
  --max-border-edge-density 0.38 \
  --max-foreground-components 6 \
  --min-largest-foreground-component-ratio 0.50 \
  --allow-invalid

python3 tools/make_teacher_category_contact_sheet.py "$FOCUS_QUALITY_DIR" --image-size 128 --samples-per-key 6

TRAIN_OUT="out/tiny_train_watch75_v7_base40_focus24_hybrid_problem3_h${HIDDEN}_l64_128_s${TRAIN_STEPS}"
mkdir -p "$TRAIN_OUT"

if [[ -f "$PROBLEM3_DIR/metadata.jsonl" ]]; then
  python3 tools/train_tiny_coordinate_mlp.py \
    --teacher-root "$BASE_QUALITY_NO_PROBLEM_DIR" \
    --teacher-root "$FOCUS_QUALITY_DIR" \
    --teacher-root "$PROBLEM3_DIR" \
    --teacher-root-repeat 1 \
    --teacher-root-repeat "${FOCUS_REPEAT:-2}" \
    --teacher-root-repeat 1 \
    --image-size 128 \
    --target-downsample-size 96 \
    --target-blur-radius 0.15 \
    --latent 64 \
    --hidden "$HIDDEN" \
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
    --out-dir "$TRAIN_OUT" \
    --out-json "$TRAIN_OUT/tiny_weights.json" \
    --out-swift "$TRAIN_OUT/TinyWeights.swift" \
    --out-bin "$TRAIN_OUT/TinyWeights.bin"
else
  python3 tools/train_tiny_coordinate_mlp.py \
    --teacher-root "$BASE_QUALITY_DIR" \
    --teacher-root "$FOCUS_QUALITY_DIR" \
    --teacher-root-repeat 1 \
    --teacher-root-repeat "${FOCUS_REPEAT:-2}" \
    --image-size 128 \
    --target-downsample-size 96 \
    --target-blur-radius 0.15 \
    --latent 64 \
    --hidden "$HIDDEN" \
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
    --out-dir "$TRAIN_OUT" \
    --out-json "$TRAIN_OUT/tiny_weights.json" \
    --out-swift "$TRAIN_OUT/TinyWeights.swift" \
    --out-bin "$TRAIN_OUT/TinyWeights.bin"
fi

printf "v7_pipeline_done %s\n" "$(date -Is)"
printf "train_out=%s\n" "$TRAIN_OUT"
