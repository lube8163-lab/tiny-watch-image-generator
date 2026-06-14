# Open Caption Mining For Broad Tiny-Image Teacher Data

This is the broad image+short-caption route. It complements both:

- iPhone SDXL teacher data: controlled, high-quality core data.
- `tools/mine_dataset_with_siglip2.py`: fixed 16-category open-dataset mining.

Use this route when the goal is thousands to 10k examples with short prompts such as `cake`, `red car`, `bicycle`, `picture frame`, or `bronze sculpture`.

There are now two caption simplification modes:

- default noun-centric mode: trims actions and relations so Open Images object labels stay clean.
- `--preserve-modifiers`: keeps short adjective/action phrases such as `red car`, `dog running`, and `small white dog sitting`. Use this for CommonCatalog-style caption diversity and prompt-composition training.

## Recommended Sources

### Open Images bbox labels

Default recommendation for 128x128 tiny-image training.

- Dataset: `vikhyatk/openimages-bbox`
- Caption source: Open Images object labels from the row `objects`
- License policy: pass `--dataset-license "CC BY 2.0"` and keep source metadata in every row.
- Best use: broad, short object prompts with bbox crops.

This is more aligned with "single object, centered, readable silhouette" than generic web captions because the script can crop around the largest source bbox before resizing.

### CommonCatalog CC-BY captions

Use as a diversity supplement after object-label quality is acceptable.

- Dataset: `common-canvas/commoncatalog-cc-by`
- Caption source: `blip2_caption`, then source `caption`, then tags.
- License policy: use per-row `licensename` / `licenseurl`; rows without a license are rejected.
- Best use: broader phrases and scenes, not the first source for single-object data.

Avoid license-unclear web caption datasets for teacher data unless every row keeps a usable source license.

## Output Layout

```text
datasets/open_mined_caption_siglip2/<run_id>/
├── manifest.json
├── metadata.jsonl
├── images_256/
├── images_128/
└── reports/
    ├── contact_sheet_top_matches.png
    ├── contact_sheet_rejected.png
    └── score_summary.json
```

`metadata.jsonl` keeps:

- `source_type: "open_caption_dataset"`
- `source_dataset`
- `source_id`
- `source_url` or `source_path`
- `license`
- `caption`, `raw_caption`, `prompt`
- `image_caption_score`
- `top1_score`
- `top2_score`: maximum negative score for this broad-caption route
- `score_margin`: `image_caption_score - top2_score`
- `negative_scores`
- `accepted`, `reject_reason`
- `saved_images`
- `width`, `height`
- `created_at_unix`

For Open Images bbox rows, `extra.object_label`, `extra.object_bbox`, and `extra.object_area` are also preserved.

## Pilot Command

Start with this before any large run:

```sh
.venv/bin/python tools/mine_caption_dataset_with_siglip2.py \
  --dataset vikhyatk/openimages-bbox \
  --dataset-license "CC BY 2.0" \
  --caption-fields objects \
  --crop-bbox \
  --out-dir datasets/open_mined_caption_siglip2/pilot_openimages_objects_200 \
  --max-images 2000 \
  --target-count 200 \
  --threshold 0.03 \
  --min-quality-score -0.02 \
  --max-negative-score 0.12 \
  --min-object-area 0.04 \
  --max-per-caption 10 \
  --exclude-captions "person,man,woman,human body,human face,boy,girl,building" \
  --batch-size 4 \
  --shuffle-buffer 1000 \
  --allow-downloads \
  --force-exit
```

Review these before scaling:

```text
datasets/open_mined_caption_siglip2/pilot_openimages_objects_200/reports/contact_sheet_top_matches.png
datasets/open_mined_caption_siglip2/pilot_openimages_objects_200/reports/contact_sheet_rejected.png
datasets/open_mined_caption_siglip2/pilot_openimages_objects_200/reports/score_summary.json
```

## Scaling To 2,000-5,000

Use the same command with a larger target:

```sh
.venv/bin/python tools/mine_caption_dataset_with_siglip2.py \
  --dataset vikhyatk/openimages-bbox \
  --dataset-license "CC BY 2.0" \
  --caption-fields objects \
  --crop-bbox \
  --out-dir datasets/open_mined_caption_siglip2/openimages_objects_2500 \
  --max-images 30000 \
  --target-count 2500 \
  --threshold 0.03 \
  --min-quality-score -0.02 \
  --max-negative-score 0.12 \
  --min-object-area 0.04 \
  --max-per-caption 80 \
  --exclude-captions "person,man,woman,human body,human face,boy,girl,building,poster" \
  --batch-size 4 \
  --shuffle-buffer 5000 \
  --allow-downloads \
  --force-exit
```

For a 10k run, raise `--target-count` to `10000` and `--max-images` to `100000` or more. Keep `--shuffle-buffer` enabled so early dataset ordering does not dominate the result.

If Hugging Face rate limits appear, set `HF_TOKEN` in the shell before running. Do not switch to license-unclear datasets just to increase throughput.

If the run fails before writing metadata with errors like `Cannot send a request, as the client has been closed`,
rerun with `--shuffle-buffer 0`. Large shuffle buffers can force the streaming loader to fetch distant parquet shards
concurrently, which is more fragile on an unstable connection. The output order is less random, but `--max-per-caption`
still prevents a few labels from fully dominating the accepted set.

## CommonCatalog Diversity Command

Run this only after object-label output looks good. For prompt-composition training, prefer `--preserve-modifiers` so colors, adjectives, and actions are not stripped from captions:

```sh
.venv/bin/python tools/mine_caption_dataset_with_siglip2.py \
  --dataset common-canvas/commoncatalog-cc-by \
  --caption-fields blip2_caption,caption,usertags \
  --out-dir datasets/open_mined_caption_siglip2/commoncatalog_captions_1000 \
  --max-images 10000 \
  --target-count 1000 \
  --threshold 0.04 \
  --min-quality-score 0.00 \
  --max-negative-score 0.10 \
  --preserve-modifiers \
  --max-per-caption 20 \
  --exclude-captions "person,man,woman,human body,human face,boy,girl,building,poster,text,logo" \
  --batch-size 4 \
  --shuffle-buffer 5000 \
  --allow-downloads \
  --force-exit
```

Expect CommonCatalog quality to be more variable. Keep it as diversity data and cap its share if the contact sheet shows scenes, text, or abstract captions.

## Mixing Policy

For the first mixed training experiment:

- iPhone SDXL: 2,000-5,000 images
- Fixed 16-category SigLIP2 mined data: category coverage and compatibility
- Broad caption/object-label mined data: additional prompt and object diversity

Start around 50% iPhone SDXL and 50% mined data. If mined contact sheets are noisy, shift back toward iPhone SDXL. Preserve `source_type`, `source_dataset`, `license`, and `score_margin` in the final mixed manifest so weak sources can be filtered later.

## Compositional Prompt Training

Use the compositional encoder when mixing noun, adjective, action, and style prompts:

```sh
.venv/bin/python tools/train_tiny_coordinate_mlp.py \
  --prompt-encoder compositional_v1 \
  --teacher-root datasets/open_mined_caption_siglip2/openimages_objects_10000 \
  --teacher-root datasets/open_mined_caption_siglip2/commoncatalog_captions_1000 \
  --image-size 32 \
  --latent 48 \
  --hidden 1024 \
  --coord-frequencies 1,2,4,8 \
  --steps 40000 \
  --preview-prompts-file configs/prompt_eval_suite.json \
  --out-dir out/tiny_train_compositional_v1
```

The default `hash_v1` encoder remains available for reproducing older toy-model runs. New broad-prompt experiments should use `compositional_v1`.

## Filtering Noisy Caption Runs

CommonCatalog can produce useful modifier/action captions, but broad caption rows are noisy. Filter mined outputs before mixing them into tiny-model training:

```sh
.venv/bin/python tools/filter_caption_dataset.py \
  --input-root datasets/open_mined_caption_siglip2/commoncatalog_modifiers_known_200 \
  --out-dir datasets/open_mined_caption_siglip2/commoncatalog_modifiers_known_compositional \
  --require-known-subject \
  --require-compositional-signal \
  --min-caption-score 0.045 \
  --min-quality-score 0.01 \
  --max-negative-score 0.09 \
  --exclude-caption-fragments "person,man,woman,baby,girl,boy,people,human,poster,logo,text,photo by,diagram,airport,california,cuba,bulgaria,stockholm,ada,astrodeep,glenridding,cortes,confusion,building,sign,recipe,gun"
```

This keeps rows such as `red car parked in parking lot`, `white rabbit eating flower`, and `cat sleeping on chair`, while removing many place names, people-centric captions, and abstract captions.
