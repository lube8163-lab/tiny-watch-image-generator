# Project Tracks

This repository has three related but separate tracks. Keeping them conceptually
separate makes old experiments easier to understand without moving files around
and risking Xcode project breakage.

## 1. Original MLP Track

Purpose:

- prove a tiny prompt-to-image UI can run on Apple Watch without Core ML;
- provide a fast Swift-only baseline for prompt normalization and contact-sheet
  tooling;
- keep a no-download demo available.

Primary files:

- `watchos_example/TinyImageWatchApp/`
- `Sources/TinyWatchGenerator/`
- `Sources/TinyPreview/`
- `Sources/TinyWatchEval/`
- `configs/prompt_eval_suite.json`
- `tools/make_watch_eval_contact_sheet.py`
- `tools/make_watch_postprocess_compare.py`
- `tools/train_tiny_coordinate_mlp.py`
- `tools/generate_weights.py`

Status:

- useful as a lightweight demo and historical baseline;
- not the current quality direction.

## 2. Core ML Stress / Probe Track

Purpose:

- test what Apple Watch can load and predict with Core ML;
- measure memory ceilings before wiring models into a generation pipeline;
- isolate risky components such as the CLIP text encoder.

Primary files:

- `watchos_example/WatchStressTestApp/`
- `watchos_example/WatchStressTestApp/Models/`
- `watchos_example/TinyImageWatchApp.xcodeproj/xcshareddata/xcschemes/WatchStressTestApp.xcscheme`
- `watchos_example/TinyImageWatchApp.xcodeproj/xcshareddata/xcschemes/WatchTextEncoderSmokeApp.xcscheme`
- `schemes/watch_sd_quantization/`
- `docs/watch/text_encoder_smoke.md`

Status:

- probe and diagnostic layer;
- keep it separate from the product-like WatchPipeline UI.

## 3. Diffusion / LCM256 Track

Purpose:

- current best-quality Apple Watch local text-to-image path;
- short free-text prompts via transient on-device CLIP text encoding;
- streamed LCM generation with CPU-only Core ML.

Primary files:

- `watchos_example/WatchPipelineSmokeApp/`
- `watchos_example/WatchPipelineSmokeApp/LCM256Assets/lcm_scheduler.json`
- `watchos_example/WatchPipelineSmokeApp/TextEncoderAssets/`
- `watchos_example/WatchPipelineSmokeApp/Models/`
- `tools/watch_lcm256_quality_eval.py`
- `tools/verify_watch_pipeline_smoke.py`
- `configs/watch_lcm256_quality_prompts.json`
- `docs/watch/watch_256_baseline_summary_2026-06-23.md`
- `docs/watch/mac_quality_eval_full_summary_2026-06-24.md`

Status:

- adopted baseline after successful 256px physical Watch tests;
- Mac quality eval is suitable for broad ranking;
- physical Watch is still required for runtime, memory, thermal, and UX checks.

## Artifact Policy

Keep in Git:

- source code,
- Xcode schemes/project wiring,
- small JSON/bin prompt assets,
- tokenizer metadata,
- scripts,
- docs.

Keep out of Git:

- generated images and contact sheets,
- downloaded model snapshots,
- converted `.mlpackage` files,
- compiled `.mlmodelc` directories,
- datasets and training outputs.

Ignored local directories:

- `datasets/`
- `dist/`
- `models/`
- `out/`
- `reports/`
