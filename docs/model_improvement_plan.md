# Model improvement plan

The current iOS Core ML path is now useful enough to diagnose model quality:

- VAE 4-bit and VAE FP16 produce nearly the same structure.
- Higher guidance reduces noise, but prompt alignment remains weak.
- The main bottleneck is likely the denoiser/text-conditioning quality, not the iOS image decoder.

## Step 1: Mac reference for the current small model

Generate a Mac-side reference sheet with the same prompt presets, seeds, steps, and guidance values used on device:

```sh
.venv/bin/python tools/generate_student_reference_grid.py \
  --candidate segmind_tiny_sd \
  --local-files-only \
  --limit 8 \
  --seeds 0 \
  --guidance-scales 4,8,10 \
  --steps 30 \
  --width 128 \
  --height 128 \
  --out-dir reports/student_reference/segmind_tiny_sd
```

Compare `reports/student_reference/segmind_tiny_sd/contact_sheet.png` with iPhone screenshots.

If the Mac reference is also weak, the model/checkpoint is the bottleneck. If the Mac reference is much better, the custom Core ML sampler still differs from the PyTorch path.

Verified on the current M2 / 8 GB Mac:

- `segmind_tiny_sd` at `128x128`, `30` steps, `guidance=4,8,10`, and eight presets completed successfully.
- Output: `reports/student_reference/segmind_tiny_sd/contact_sheet.png`
- The Mac reference also collapses to similar-looking shapes across prompts, so prompt alignment is already weak before the iOS/Core ML path.

## Step 2: Mac SDXL teacher dataset

Use Mac-side SDXL generation for small teacher-data probes so long runs do not
stress the iPhone. The local `SDXL_test` workspace has an SDXL base cache at:

```text
/Users/tasuku/Desktop/XcodeProjects/SDXL_test/.cache/huggingface/hub
```

`/Users/tasuku/Desktop/SDXL_test` is also accepted; the script resolves the
symlink and uses the same cache. The SD1.5 teacher dataset app is useful as a
reference for the prompt catalog and output layout, but the desktop app target
itself is currently SD1.5-oriented. The practical path is the Python/Diffusers
SDXL generator below.

Smoke test:

```sh
.venv/bin/python tools/generate_sdxl_teacher_dataset.py \
  --limit 1 \
  --variants-per-prompt 1 \
  --seeds 0 \
  --steps 1 \
  --width 256 \
  --height 256 \
  --target-sizes 128,64 \
  --out-dir datasets/sdxl_mac_teacher_smoke \
  --overwrite \
  --local-files-only
```

Important MPS note:

- Do not enable `--attention-slicing` on the current M2 / 8 GB Mac. In testing,
  SDXL base at `384x384` produced non-finite latents on the first denoising
  step, and all saved images became pure black.
- `tools/generate_sdxl_teacher_dataset.py` now defaults attention slicing off
  and aborts if a generated image has near-zero dynamic range.
- `datasets/sdxl_mac_teacher_fixed16_v1` from the interrupted run contains 62
  black apple images and should not be used for training.

Local probe run:

```sh
.venv/bin/python tools/generate_sdxl_teacher_dataset.py \
  --limit 16 \
  --variants-per-prompt 1 \
  --seeds 0 \
  --steps 4 \
  --width 384 \
  --height 384 \
  --target-sizes 128,64 \
  --out-dir datasets/sdxl_mac_teacher_probe
```

Output layout:

```text
datasets/sdxl_mac_teacher_probe/
├── images_128/
├── images_64/
├── metadata.jsonl
├── contact_sheet.png
└── manifest.json
```

The metadata separates the short learner condition from the verbose SDXL prompt:

- `prompt` / `conditioning_prompt`: compact training condition such as `cat`,
  `cat front view`, or `toy cat`.
- `teacher_prompt`: full SDXL prompt used for image generation.

This proves local prompt/image generation works, but it is slow on the current
Mac path. Verified timings:

- `steps=1`, `256x256`, one image: about 74 seconds end-to-end.
- `steps=1`, `256x256`, two images in one process: about 41 seconds for the first image and 50 seconds for the second image.
- After adding SDXL teacher metadata/output handling, `steps=1`, `256x256`, one
  image completed successfully in about 148 seconds end-to-end on the current
  Mac/MPS environment.
- With attention slicing disabled, `steps=1`, `384x384`, one image completed
  successfully without black output in about 158 seconds end-to-end.
- With attention slicing disabled, `steps=1`, `256x256`, one image completed in
  about 178 seconds end-to-end. Reducing from `384` to `256` did not materially
  improve throughput on this MPS path.

Use this local path only for smoke tests and tiny probes. At roughly 2 to 3
minutes per denoising step, a `384x384`, `8` step, `1024` image run can take
around two weeks. Bulk SDXL teacher generation should move to Core ML on Mac if
we build a CLI around the existing compiled resources, or to a cloud GPU.

If running a small local probe:

```sh
caffeinate -dimsu .venv/bin/python tools/generate_sdxl_teacher_dataset.py \
  --limit 16 \
  --variants-per-prompt 1 \
  --seeds 0 \
  --steps 4 \
  --width 384 \
  --height 384 \
  --target-sizes 128,64 \
  --out-dir datasets/sdxl_mac_teacher_probe
```

Before running a larger batch, quit Xcode, Simulator, browser tabs, and other memory-heavy apps. This will not make the Mac behave like the iPhone Core ML/ANE path, but it reduces swapping and avoids the worst slowdowns on an 8 GB machine.

For a cloud GPU run, use 768px SDXL teacher images and save 256/128/64
derivatives. A 24 GB RTX 4090 handled `BATCH_SIZE=2` reliably:

```sh
bash tools/cloud_generate_sdxl_fixed16.sh
```

After the fixed16 dataset validates, expand to the action/modifier/composition
set:

```sh
RUN_NAME=sdxl_cloud_teacher_expanded32_v2_pilot \
LIMIT=8 \
SEEDS=0,1 \
bash tools/cloud_generate_sdxl_expanded_v2.sh
```

If the pilot looks good:

```sh
bash tools/cloud_generate_sdxl_expanded_v2.sh
```

The first expanded32 cloud run is available locally:

```text
datasets/sdxl_cloud_teacher_expanded32_v2
```

This full set validates structurally, but visual review found residual
repeated-object/contact-sheet artifacts. Use the stricter filtered subset as the
default controlled teacher root:

```text
datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict
```

It contains 1730 valid images across 32 categories after excluding the most
artifact-prone seed/variant combinations (`seed=7`, `v03`, `c02`, `a02`). The
full set should be kept as source material, but not used as the primary training
root until additional manual review or category-specific regeneration is done.

## Step 3: Training direction

Do not fine-tune directly on broad prompts first. Start with a narrow target:

- 4 to 8 classes: `cat`, `dog`, `sun`, `moon`, `flower`, `car`, `face`, `house`
- 4 to 8 seeds per class
- 128x128 targets
- fixed simple style prompts

The first training target should be a small denoiser/text-conditioning improvement, not the VAE. The VAE comparison suggests that decoder precision is not the main quality limiter.

## Step 4: Broad Prompt Composition

For the pure Swift `TinyCoordinateMLP` track, the main generalization bottleneck is prompt representation. The old `hash_v1` encoder maps the canonicalized prompt to one opaque random vector, so `cat`, `red cat`, and `cat sitting` do not share useful structure unless they collapse to the same key.

Use `compositional_v1` for new broad-prompt runs. It keeps the seed noise but adds deterministic features for:

- subject aliases such as `cat`, `car`, `flower`
- colors such as `red`, `blue`, `white`
- actions such as `sitting`, `running`, `flying`
- size/style modifiers such as `small`, `cute`, `anime`, `watercolor`
- short unknown tokens and bigrams as a fallback

Keep the evaluation prompts fixed with:

```sh
--preview-prompts-file configs/prompt_eval_suite.json
```

Recommended first broad run:

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

If CommonCatalog contact sheets are noisy, cap or remove that root and train on Open Images object captions plus a smaller SDXL/iPhone synthetic set for color/action prompts.

Latest broad-prompt experiment:

```text
out/tiny_train_compositional_core40_mod67x6_seednorm_h1024_l48_32
```

Settings:

- Open Images caption data capped to core subject keys, max 80 samples per key.
- Filtered CommonCatalog modifier/action subset repeated 6x.
- `prompt_encoder=compositional_v1`
- `image_size=32`, `latent=48`, `hidden=1024`, `steps=40000`
- final loss around `0.0043`
- output weight binary around `1.1 MB`

Result: the lower loss improves reconstruction and some color/modifier differences are visible, but Open Images real-image crops still push the tiny coordinate MLP toward texture/background averages. For the next quality step, use controlled simple teacher images as the primary source and keep Open Images/CommonCatalog as supplemental diversity only.

Recommended next dataset mix:

- 70-85% controlled centered teacher images from
  `datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict`, simple background.
- 10-20% Open Images object crops, capped per key.
- 5-10% filtered CommonCatalog modifier/action rows.

Do not make broad Open Images captions the primary source for the pure Swift MLP. They are useful for vocabulary coverage, but they are too visually noisy for 32x32 coordinate-regression training.

Latest SDXL controlled-teacher experiment:

```text
out/tiny_train_sdxl32_balanced24_h1536_l48_128
```

Settings:

- Source dataset: `datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict_balanced24`
- 32 categories, 24 images per category, 768 images total
- `prompt_encoder=compositional_v1`
- `image_size=128`, `latent=48`, `hidden=1536`, `steps=12000`
- final loss around `0.0067`
- output weight binary around `2.4 MB`
- active app weights and Watch generation size were updated from this run

Comparison:

- Full curated-strict, 1730 images, 64px, 12000 steps: final loss around `0.0088`.
- Balanced24, 768 images, 64px, 12000 steps: final loss around `0.0060`.
- Balanced24, 768 images, 64px, hidden 1536, 12000 steps: final loss around `0.0038`.
- Balanced24, 768 images, 128px, hidden 1536, 12000 steps: final loss around `0.0067`.

Result: for the current 1.1 MB coordinate MLP, simply increasing controlled
teacher image count causes visible averaging. A smaller, balanced, higher-quality
subset fits the model better. Increasing hidden size from 1024 to 1536 clearly
improves shape retention and remains small enough for the current Watch runtime.
128px output is smoother and can preserve more fine structure, but Watch runtime
cost is expected to be roughly 4x the 64px path. Before generating tens of
thousands of images, keep dataset growth tied to model capacity; otherwise the
extra data is likely to blur together.

Current coverage gap:

- The Watch UI has prompts that the controlled SDXL set did not include:
  `fox`, `owl`, `banana`, `orange`, `strawberry`, `cake`, `pizza`, `bread`,
  `mountain`, `cloud`, `clock`, `ball`, `guitar`, `shoe`.
- `configs/sdxl_tiny_teacher_prompts_v3_watch.json` adds those categories and
  keeps the existing 32-category set, for 46 categories total.
- `tools/cloud_generate_sdxl_watch_v3.sh` is the next cloud-generation entry
  point. Its default `SEEDS=0,1,2,5` produces 2208 planned rows before rejects.

## Notes on SDXL Core ML resources

`SDXL_test/artifacts/sdxl-base-ios/Resources` contains compiled Core ML SDXL resources. The Apple Python pipeline can load compiled `.mlmodelc` resources, but its default loader expects `Unet.mlmodelc`; the local SDXL resources are chunked as `UnetChunk1.mlmodelc` and `UnetChunk2.mlmodelc`. Using those from Python would need an additional chunk-aware wrapper. The current dataset script therefore uses the local Diffusers cache first.

The Swift `StableDiffusionSample` CLI does support `--xl` and chunked UNet resources, and it builds successfully from `SDXL_test/.build/ml-stable-diffusion-1.1.1`. A first Mac run with the compiled SDXL resources did not produce output after several minutes and was stopped. The iPhone path can still be much faster because it runs the Core ML graph through the iPhone-oriented runtime/ANE path.
