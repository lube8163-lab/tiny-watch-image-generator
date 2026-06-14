# Open Dataset Mining For Mixed Teacher Data

This workflow treats open-dataset mining as supplemental data for the iPhone SDXL teacher dataset.

For broad, non-fixed-category short captions, use the newer workflow in
`docs/open_caption_mining.md` and `tools/mine_caption_dataset_with_siglip2.py`.
That route can use Open Images bbox object labels as few-word prompts and crop around
source bboxes before saving 256/128 images.

## Role In The Mixed Dataset

- iPhone SDXL data is the controlled, high-quality core dataset.
- SigLIP2-mined open data is supplemental diversity data.
- Start with a conservative mixture:
  - iPhone SDXL: 30-50%
  - SigLIP2-mined open data: 50-70%
- For the first training experiment:
  - iPhone SDXL: 2,000-5,000 images
  - SigLIP2 mined: 2,000-5,000 images
  - Total: 4,000-10,000 images

If mined quality is unstable, shift the mixture back toward iPhone SDXL data before increasing scale.

## Primary 16 Categories

The mining config is aligned to the current iPhone SDXL category set:

```text
apple
bird
car
castle
cat
dog
face
fish
flower
house
moon
robot
star
sun
train
tree
```

## Scoring Policy

The miner evaluates every image against all category prompts with `google/siglip2-base-patch16-224`.

Each metadata row includes:

- `top1_score`
- `top2_score`
- `score_margin`
- `negative_scores`
- `accepted`
- `reject_reason`

Reject reasons include:

- `missing_license`
- `low_top1_score`
- `low_score_margin`
- `center_class_mismatch`
- `low_center_score`
- `low_center_margin`
- `high_negative_score`
- `class_topk_overflow`

Negative concepts currently scored:

- `text`
- `logo`
- `watermark`
- `crowded_scene`
- `multiple_objects`
- `human_group`
- `blurry_image`

## Output Layout

```text
datasets/open_mined_siglip2/<run_id>/
├── manifest.json
├── metadata.jsonl
├── images_256/
├── images_128/
└── reports/
    ├── contact_sheet_top_matches.png
    ├── contact_sheet_rejected.png
    └── score_summary.json
```

## Recommended Pilot Source

Use Open Images bbox through the Hugging Face mirror when category balance matters:

```sh
.venv/bin/python tools/mine_dataset_with_siglip2.py \
  --dataset vikhyatk/openimages-bbox \
  --split train \
  --dataset-license "CC BY 2.0" \
  --max-images 512 \
  --top-k-per-class 20 \
  --threshold 0.03 \
  --min-margin 0.01 \
  --min-center-score 0.02 \
  --min-center-margin 0.005 \
  --max-negative-score 0.09 \
  --metadata-bonus 0.03 \
  --batch-size 4 \
  --out-dir datasets/open_mined_siglip2/pilot_openimages_bbox_16cat
```

This mirror exposes object labels, which makes it better for class-aligned mining than random URL-only Open Images mirrors.

License note: Open Images images are listed as CC BY 2.0 and annotations as CC BY 4.0, but the original dataset warns users to verify image licenses when needed. Keep `source_dataset`, `source_path`, and `license` in every row.

## Current Pilot Result

Run:

```text
datasets/open_mined_siglip2/pilot_openimages_bbox_16cat_20260606
```

Summary:

- scanned: 512
- accepted: 142
- rejected: 370
- full 20-image pilot buckets reached:
  - `flower`
  - `car`
  - `house`
  - `face`
- partial buckets:
  - `castle`: 14
  - `bird`: 9
  - `apple`: 9
  - `robot`: 9
  - `train`: 5
  - `dog`: 4
  - `tree`: 4
  - `sun`: 3
  - `cat`: 2
  - `moon`: 2
  - `fish`: 1
  - `star`: 0

This is enough to validate the mixed-schema output, but not enough for a balanced 16-category training set.

## Next Scaling Step

For unattended mining, prefer the automation runner. It launches multiple shards, resumes existing shard outputs, merges accepted rows into one final dataset directory, and stops when every category reaches the target count or `--max-shards` is exhausted.

```sh
.venv/bin/python tools/run_open_dataset_mining_auto.py \
  --run-id openimages_bbox_auto_20_each \
  --dataset vikhyatk/openimages-bbox \
  --split train \
  --dataset-license "CC BY 2.0" \
  --target-per-class 20 \
  --images-per-shard 512 \
  --max-shards 12 \
  --threshold 0.03 \
  --min-margin 0.01 \
  --min-center-score 0.02 \
  --min-center-margin 0.005 \
  --max-negative-score 0.09 \
  --metadata-bonus 0.03 \
  --batch-size 4
```

The final combined output is written directly under:

```text
datasets/open_mined_siglip2/openimages_bbox_auto_20_each/
```

Shard internals are kept under `_shards/`. The final root contains:

- `manifest.json`
- `automation_status.json`
- `metadata.jsonl`
- `images_128/`
- `images_256/`
- `reports/contact_sheet_top_matches.png`
- `reports/contact_sheet_rejected.png`
- `reports/score_summary.json`

Use `--local-files-only` by default. The current workspace already has the Hugging Face cache for `google/siglip2-base-patch16-224`. If the cache is missing, use `--allow-downloads` once.

Use contact sheets before mixing into training. For categories that remain sparse (`star`, `moon`, `sun`, `fish`, `robot`), prefer iPhone SDXL generation or category-specific open sources rather than forcing ambiguous mined images.

## Automation Notes

The app project at `/Users/tasuku/Desktop/SemanticCompression-v2` contains SigLIP2-related Core ML/tagging resources, but the mining pipeline uses the Hugging Face PyTorch model because it needs image-text prompt similarity against arbitrary class prompts.

The local HF cache currently has:

```text
~/.cache/huggingface/hub/models--google--siglip2-base-patch16-224
```

The automation runner calls:

```text
tools/mine_dataset_with_siglip2.py
```

with `--force-exit` so Hugging Face streaming subprocesses do not linger after outputs are written.

Resume behavior:

- Existing `_shards/shard_XXX/manifest.json` means that shard is reused.
- Delete one shard directory to rerun only that shard.
- Delete the whole run directory to start clean.
- `automation_status.json` records the latest accepted counts and missing categories.

## Caption And Label Prefiltering

Some sources include useful text metadata:

- `common-canvas/commoncatalog-cc-by`: `title`, `caption`, `blip2_caption`, tags
- `vikhyatk/openimages-bbox`: object labels

Use `--metadata-prefilter` to skip rows whose text metadata does not mention any target category aliases before image decoding and SigLIP2 scoring:

```sh
.venv/bin/python tools/run_open_dataset_mining_auto.py \
  --run-id openimages_bbox_prefilter_20_each \
  --dataset vikhyatk/openimages-bbox \
  --split train \
  --dataset-license "CC BY 2.0" \
  --target-per-class 20 \
  --images-per-shard 512 \
  --max-shards 12 \
  --metadata-prefilter \
  --max-source-rows-per-shard 3000 \
  --batch-size 4
```

This is useful when the source has reliable captions or object labels. It reduces wasted image decoding and scoring.

Do not rely on this alone for sparse or symbolic categories. `moon`, `star`, and sometimes `sun` may be absent from captions/object labels even when the image visually contains them, so these categories should still be covered by iPhone SDXL or category-specific open sources.

## Mixing Procedure

1. Generate or collect iPhone SDXL data into its normal teacher dataset directory.
2. Mine open data with this script.
3. Review:
   - `reports/contact_sheet_top_matches.png`
   - `reports/contact_sheet_rejected.png`
   - `reports/score_summary.json`
4. Use only rows with `accepted: true`.
5. Cap per-category mined rows so the mined set does not dominate weak categories.
6. Build the training manifest with a target mixture:
   - first experiment: 50% iPhone SDXL, 50% mined
   - if mined images look strong: up to 30-50% iPhone and 50-70% mined
   - if mined images look noisy: return to 60-80% iPhone SDXL

Keep source metadata in the final mixed manifest so later training runs can be filtered by `source_type`.
