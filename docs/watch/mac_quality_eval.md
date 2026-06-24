# Mac Quality Eval For Watch LCM256

The Apple Watch product path still needs physical-device checks for runtime,
memory peak, thermal behavior, and UX. Broad model-quality sweeps can run on Mac
because they do not need the Watch hardware loop.

## Repository Split

Keep the eval harness in this repository for now:

- prompt suites, scripts, and docs stay versioned next to the model assets they
  evaluate;
- generated PNGs, manifests, and contact sheets stay under ignored `reports/`;
- commits can tie a model change and its exact eval recipe together.

Move this into a separate repository only when the generated dataset/reports need
to be published independently, shared with non-app collaborators, or reused
across multiple model repos.

## Broad 256px Run

Default command:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py
```

That uses:

- `configs/watch_lcm256_quality_prompts.json`
- `watchos_example/WatchPipelineSmokeApp/LCM256Assets/lcm_scheduler.json`
- `watchos_example/WatchPipelineSmokeApp/TextEncoderAssets/clip_vocab.json`
- `watchos_example/WatchPipelineSmokeApp/TextEncoderAssets/clip_merges.txt`
- `dist/lcm_dreamshaper_v7/text_encoder_probe/clip_text_encoder_77.mlpackage`
- `dist/lcm_dreamshaper_v7/unet_32x32_6bit.mlpackage`
- `dist/lcm_dreamshaper_v7/vae_decoder_256x256_noattn_4bit.mlpackage`
- CPU-only Core ML execution
- guidance `6`, matching the current Watch UI default

The default suite is 74 prompts x 4 seeds = 296 images. Output goes to:

```text
reports/watch_lcm256_quality/lcm256_quality_YYYYMMDD_HHMMSS/
```

On the current Mac setup, the 296-image full suite completed in about `35.7`
minutes after Core ML cache warmup, averaging `7.24s/image`. Treat this as a
batch job, but it is short enough to run whenever a candidate model or prompt
change looks promising.

Important files:

- `report.md`: aggregate summary and caveats.
- `manifest.json`: full run metadata and result records.
- `manifest.jsonl`: one record per image.
- `contact_sheets/overview.png`: up to the first 96 generated images.
- `contact_sheets/by_genre/*.png`: review sheets grouped by genre.
- `images/`: raw 256px outputs.

## Smaller Runs

Fast smoke:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py --max-images 4
```

Single genre:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py --genres animals --seeds 1 7 24 42
```

Specific prompts:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py \
  --prompt-ids animal_tabby_cat nature_snowy_mountain relation_astronaut_riding_horse \
  --seeds 1 7 24 42
```

Fixed output directory:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py \
  --out-dir reports/watch_lcm256_quality/manual_lcm256_baseline
```

## How To Read The Report

The numeric flags are only outlier detectors:

- `low_contrast`: likely washed out or too flat.
- `very_dark` / `very_light`: exposure outliers.
- `high_clip`: many decoded channels fell outside `[-1, 1]`.
- `very_soft`: extremely low local luma variation.

These flags do not measure semantic prompt fit. The main quality decision should
still come from visually scanning the contact sheets.

Current full-run summary:

- `docs/watch/mac_quality_eval_full_summary_2026-06-24.md`
- `reports/watch_lcm256_quality/full_lcm256_g6/`

## Mac Versus Watch

This script uses the same scheduler, tokenizer, prompt expansion, Core ML text
encoder package, compressed UNet package, decoder package, seed rule, guidance,
and RGB conversion as the Watch path. It is appropriate for ranking prompts,
seeds, guidance values, and candidate assets.

It is not a bit-for-bit Watch replica. Python Core ML loads `.mlpackage` assets,
while the Watch app loads compiled streamed `.mlmodelc` chunks. Final adoption
still requires an Apple Watch run for memory and runtime confirmation.
