# Codex instructions: iPhone teacher dataset generation

Use this document as the starting brief for a new Codex thread in:

```text
/Users/tasuku/Desktop/XcodeProjects/SDXL_test
```

Codex can inspect local project files on disk, but cannot read prior Codex chat history unless it was written into project files. Treat this document and the repository as the source of truth.

## Goal

Build an iPhone-side teacher dataset generator for the existing SDXL Core ML test app.

The target workflow is:

1. Keep the SDXL pipeline loaded on iPhone.
2. Generate images continuously from a prompt/seed schedule.
3. Save each generated image plus its prompt/settings metadata immediately.
4. Export the resulting dataset from the app's Documents folder.
5. Use the dataset later to train or distill a much smaller 128x128 image generator.

The reason for using iPhone is performance. On this Mac, Diffusers/MPS SDXL generation is very slow. The existing iPhone app has already shown roughly 768x768, 20 steps in about 20 seconds, so the iPhone Core ML/ANE path is likely the best local teacher-data producer.

## Current project facts

- App target: `SDXLCoreMLTest`
- Main app code:
  - `SDXLCoreMLTest/SDXLGeneratorViewModel.swift`
  - `SDXLCoreMLTest/ContentView.swift`
- Local Stable Diffusion package:
  - `Vendor/StableDiffusionLocal`
- Existing model resources:
  - `artifacts/sdxl-base-ios/Resources`
  - Contains `TextEncoder.mlmodelc`, `TextEncoder2.mlmodelc`, `UnetChunk1.mlmodelc`, `UnetChunk2.mlmodelc`, `VAEDecoder.mlmodelc`, `vocab.json`, `merges.txt`
- Existing runtime already supports chunked SDXL UNet through `StableDiffusionXLPipeline(resourcesAt:configuration:reduceMemory:)`.
- Existing app already caches the pipeline in `SDXLPipelineCache`.
- Existing generation defaults are roughly:
  - SDXL
  - 768x768
  - 20 steps
  - DPMSolverMultistep
  - guidance around 5
  - reduce memory enabled

## Recommended approach

Do not start by converting a 256x256 model.

First, use the existing 768x768 model and add a batch dataset mode. Save the full 768 image and also save downsampled teacher targets, for example 256x256 and 128x128. This is more likely to preserve prompt quality than native 256 SDXL generation, and it avoids a long conversion step before validating the dataset workflow.

Only after the batch workflow is stable should we consider converting lower-resolution resources.

## Dataset output format

Write datasets under the app's Documents directory:

```text
Documents/TeacherDatasets/<run_id>/
├── manifest.json
├── metadata.jsonl
├── images_768/
│   ├── cat_v0_seed000000.png
│   └── ...
├── images_256/
│   ├── cat_v0_seed000000.png
│   └── ...
└── images_128/
    ├── cat_v0_seed000000.png
    └── ...
```

Each `metadata.jsonl` line should include:

```json
{
  "id": "cat_v0_seed000000",
  "key": "cat",
  "title": "Cat",
  "prompt": "cute cat sitting, clean anime illustration, simple background",
  "negative_prompt": "low quality, blurry, distorted, deformed, bad anatomy",
  "seed": 0,
  "steps": 20,
  "guidance_scale": 5.0,
  "scheduler": "dpmSolverMultistep",
  "source_width": 768,
  "source_height": 768,
  "saved_images": {
    "768": "images_768/cat_v0_seed000000.png",
    "256": "images_256/cat_v0_seed000000.png",
    "128": "images_128/cat_v0_seed000000.png"
  },
  "elapsed_seconds": 20.4,
  "created_at_unix": 1780000000
}
```

`manifest.json` should summarize:

- run id
- app version/build
- model family and resource directory
- resolution
- total planned jobs
- completed jobs
- started/ended timestamps
- prompt set version
- generation settings
- device name if easy to collect

## Prompt schedule

Start small. Use 4 to 8 simple classes:

- `cat`
- `dog`
- `sun`
- `moon`
- `flower`
- `car`
- `face`
- `house`

Use simple English prompts. SDXL 1.0 is more reliable with English prompts.

Example prompt variants:

```json
[
  {
    "key": "cat",
    "title": "Cat",
    "prompt": "cute cat sitting, centered subject, simple clean background, readable silhouette"
  },
  {
    "key": "dog",
    "title": "Dog",
    "prompt": "cute dog sitting, centered subject, simple clean background, readable silhouette"
  },
  {
    "key": "sun",
    "title": "Sun",
    "prompt": "bright sun in a blue sky, centered subject, simple clean background, readable silhouette"
  }
]
```

Generate multiple seeds per prompt. A good first pilot:

- 4 prompts
- 2 seeds per prompt
- 20 steps
- 768 source saved with 256 and 128 downsampled targets

After validating export and metadata, scale to:

- 8 prompts
- 8 to 32 seeds per prompt
- 20 steps

## Implementation tasks

1. Add dataset models:
   - `TeacherPrompt`
   - `TeacherGenerationJob`
   - `TeacherDatasetManifest`
   - `TeacherDatasetRunState`

2. Add a batch runner:
   - Prefer an `actor TeacherDatasetRunner` or a serial queue.
   - Reuse `SDXLPipelineCache` so the model is loaded once.
   - Run one generation at a time.
   - Save each image and metadata line before moving to the next job.
   - Support pause/stop.

3. Add image saving helpers:
   - Save original `UIImage` as PNG.
   - Downsample to 256 and 128 using CoreGraphics/UIGraphicsImageRenderer.
   - Avoid keeping all generated images in memory.

4. Add UI:
   - A settings/batch screen is enough.
   - Controls:
     - output run name
     - prompt count or prompt set
     - seeds per prompt
     - steps
     - guidance
     - start / pause / stop
   - Show:
     - current job
     - completed / total
     - last elapsed time
     - output folder path

5. Keep the app awake:
   - Set `UIApplication.shared.isIdleTimerDisabled = true` while a batch is running.
   - Restore it when stopped or completed.
   - Document that iOS will not run this reliably in the background. The app should stay foregrounded.

6. Handle thermals and battery:
   - Monitor `ProcessInfo.processInfo.thermalState`.
   - Auto-pause or warn on `.serious` or `.critical`.
   - Consider warning if battery is low and not charging.

7. Make files exportable:
   - Ensure the app supports file sharing / opening Documents in place if not already configured.
   - The dataset should be retrievable through Finder device file sharing.

8. Logging:
   - Print `[TeacherDataset]` lines to Xcode console.
   - Log each job start/end, seed, prompt key, elapsed time, and save paths.

## Validation checklist

Before scaling up:

1. Run a 2-image batch on iPhone.
2. Confirm the app does not reload the model for the second image.
3. Confirm files exist:
   - `manifest.json`
   - `metadata.jsonl`
   - at least one PNG in each image folder
4. Pull the dataset folder from Finder.
5. Check that metadata paths match real files.
6. Check that the 128x128 files look like valid downsampled images.
7. Leave a 10 to 20 image batch running and watch thermal behavior.

## Lower-resolution conversion plan

The existing conversion wrapper supports static latent sizes:

- `latent-h 96`, `latent-w 96` -> 768x768
- `latent-h 64`, `latent-w 64` -> 512x512
- `latent-h 32`, `latent-w 32` -> 256x256

Prefer trying 512 before 256. SDXL was not primarily trained for tiny native resolutions, so 256 native generation may lose prompt quality even if it is faster.

Example 512 conversion command:

```sh
.venv/bin/python scripts/convert_stable_diffusion.py \
  --apple-repo .build/ml-stable-diffusion-1.1.1 \
  --python-bin .venv/bin/python \
  --output-dir artifacts/sdxl-base-ios-512 \
  --cache-dir .cache/huggingface \
  --model-family sdxl \
  --model-version stabilityai/stable-diffusion-xl-base-1.0 \
  --custom-vae-version madebyollin/sdxl-vae-fp16-fix \
  --latent-h 64 \
  --latent-w 64
```

Example 256 conversion command:

```sh
.venv/bin/python scripts/convert_stable_diffusion.py \
  --apple-repo .build/ml-stable-diffusion-1.1.1 \
  --python-bin .venv/bin/python \
  --output-dir artifacts/sdxl-base-ios-256 \
  --cache-dir .cache/huggingface \
  --model-family sdxl \
  --model-version stabilityai/stable-diffusion-xl-base-1.0 \
  --custom-vae-version madebyollin/sdxl-vae-fp16-fix \
  --latent-h 32 \
  --latent-w 32
```

After conversion, verify resources:

```sh
scripts/verify_ios_resources.sh artifacts/sdxl-base-ios-512/Resources
scripts/verify_ios_resources.sh artifacts/sdxl-base-ios-256/Resources
```

If the lower-resolution resources are used in app, update `ResolutionPreset`, resource lookup paths, and dataset metadata to reflect the selected source resolution.

## Important constraints

- Do not rely on background execution. Keep the app foregrounded.
- Do not keep all generated images in RAM.
- Do not overwrite datasets. Use a unique run id.
- Do not start with a huge batch. Validate with 2 images, then 10 to 20, then scale.
- Do not assume native 256 SDXL will produce better training data than 768 downsampled output.

## Suggested first Codex task

Implement a minimal teacher dataset mode in `SDXLCoreMLTest` that:

1. Uses the existing 768x768 resources.
2. Runs a hardcoded pilot set of 4 prompts x 2 seeds.
3. Saves `images_768`, `images_256`, `images_128`, `metadata.jsonl`, and `manifest.json`.
4. Shows start/stop/progress in the UI.
5. Keeps the pipeline loaded between images.
6. Disables the idle timer while running.

After that works, generalize the prompt list and batch settings.
