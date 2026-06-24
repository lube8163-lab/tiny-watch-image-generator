# WatchPipelineSmokeApp Current Smoke Setup

Updated: 2026-06-24

This note captures the current expected WatchPipelineSmokeApp smoke-test setup and reference output.
The 256px path is now the adopted Watch diffusion baseline.

Related quality notes:

`pipeline_quality_notes.md`

Text encoder smoke notes:

`text_encoder_smoke.md`

## Default App Run

The app UI is now prompt-first. It shows a short prompt field, a Generate
button, and the generated image. Pipeline details remain in Xcode console logs.

- Prompt field default: `cat mascot`
- Prompt conditioning: current build tokenizes the typed prompt on Watch, runs
  `clip_text_encoder_77.mlmodelc` as a transient model, then releases it and
  purges the Core ML cache before streamed LCM generation.
- Short typed prompts are expanded with a small global style prior before CLIP
  encoding. For example, `spaceship banana` is conditioned as
  `spaceship banana, single subject, centered, full object visible, clean anime
  illustration, simple background`.
- After a completed run, the same Generate button is labeled `Reroll Seed` while
  the prompt is unchanged. Tapping it generates the same prompt with a fresh
  random seed.
- Preset embeddings remain bundled as a fallback and regression baseline.
- Seed default: `Random`
- Pipeline: `LCM256 6b`
- UNet: `LCM 256 6-bit 16p`
- VAE: `256 4-bit`
- Guidance: `6`
- Preview: `Smooth`

## Golden Reference Run

The checked-in pixel golden reference is still the older 128px baseline. The
current LCM256 path is validated with device logs, Mac quality contact sheets,
and bundle-shape checks. The verifier's `--family lcm256` mode checks bundle
shape but skips the 128px reference image.

- Scheme: `WatchPipelineSmokeApp`
- Pipeline: `LCM128 6b`
- UNet: `LCM 128 6-bit 16p`
- VAE: `128 4-bit`
- Preset: `Mascot Cat`
- Prompt key: `cat_mascot`
- Seed: `Curated 1`
- Guidance: `6`
- Preview: `Sharp x2`
- Expected Run ID: `cat_mascot-s1-g6-sharp2x`

Reference image:

`reports/watch_pipeline_reference/final_default_cat_mascot_s1_g6_coreml_16p/coreml.png`

Reference Watch preview image:

`reports/watch_pipeline_reference/final_default_cat_mascot_s1_g6_coreml_16p/coreml_preview_sharp2x.png`

Reference manifest:

`reports/watch_pipeline_reference/final_default_cat_mascot_s1_g6_coreml_16p/coreml.json`

Reference metrics:

- Final RMS: `0.7768`
- Decoded RMS: `0.3999`
- Clipped: `4/49152`
- Decoded shape: `1x3x128x128`

## Bundle Shape

The app target should bundle the streamed 16-part 256px LCM UNet, the 256px VAE
decoder, the shared LCM prompt assets, the 256px scheduler asset, and the
transient text encoder assets:

- `lcm_unet_32x32_6bit_16p_part1.mlmodelc` ... `lcm_unet_32x32_6bit_16p_part16.mlmodelc`
- `vae_decoder_256x256_noattn_4bit.mlmodelc`
- `TextEncoderAssets/clip_text_encoder_77.mlmodelc`
- `TextEncoderAssets/clip_vocab.json`
- `TextEncoderAssets/clip_merges.txt`
- `LCMAssets/prompt_presets.json` currently has `37` preset embeddings,
  including extra probe prompts for `dog logo`, `horse`, `astronaut`, `bird`,
  `blue bird`, and `flying blue bird`.
- `LCM256Assets/lcm_scheduler.json` has latent shape `1x4x32x32` and decoded
  shape `1x3x256x256`.

Known unwanted model families should not be in the built app bundle:

- `*8x8*.mlmodelc`
- `*16x16*.mlmodelc`
- `*24x24*.mlmodelc`

## Useful Build Check

```sh
xcodebuild -quiet \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme WatchPipelineSmokeApp \
  -destination generic/platform=watchOS \
  CODE_SIGNING_ALLOWED=NO \
  -derivedDataPath /private/tmp/watch_pipeline_check_build \
  build
```

## Useful Bundle Checks

```sh
find /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app \
  -name 'lcm_unet_*.mlmodelc'

find /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app \
  -name '*8x8*.mlmodelc'

find /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app \
  -name '*16x16_4bit*.mlmodelc'

find /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app \
  -maxdepth 4 -path '*TextEncoderAssets*'

du -sh /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app
```

Recent expected app size is about `886M`, with `TextEncoderAssets` accounting
for about `236M`.

Or run the single verifier:

```sh
./.venv/bin/python tools/verify_watch_pipeline_smoke.py \
  --family lcm256 \
  --app /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app
```

Expected verifier output includes:

```text
watch-pipeline-smoke: ok
  family: lcm256
  unet_chunks: 16
  decoder: vae_decoder_256x256_noattn_4bit.mlmodelc
  latent_shape: 1x4x32x32
  decoded_shape: 1x3x256x256
  text_encoder: TextEncoderAssets/clip_text_encoder_77.mlmodelc
  reference_checked: false
  run_id: lcm256-smoke
```

## Device Notes

- Core ML is intentionally CPU-only.
- The text encoder is loaded only during prompt conditioning, then released
  before LCM chunk generation begins.
- LCM UNet chunks are loaded and predicted one at a time during generation.
- The app purges the E5RT bundle cache before LCM load.
- Metrics include `Run ID`, `Cond`, `Prompt Key`, `Preset`, `Seed`,
  `Guidance`, `Preview`, chunk package sizes, chunk load totals, and chunk
  predict totals.
- Xcode logs include prompt input, the expanded text conditioning prompt, text
  encoder tokenizer/load/predict timing, resolved preset key for fallback/seed
  labeling, actual random seed,
  per-step chunk load/predict totals, and the final chunk timing summary. Set
  `logsDetailedLCMChunks` to `true` in `PipelineSmokeView.swift` when per-chunk
  model load/predict logs are needed.

## Device Test Checklist

Run these on the physical Apple Watch after installing the latest build:

1. Launch `WatchPipelineSmokeApp` and confirm the first screen is only the prompt
   field, Generate button, and no developer picker list.
2. Leave the default prompt as `cat mascot`, tap Generate, and confirm it
   completes without memory pressure. Console logs should include
   `resolvedPreset=cat_mascot`, `text encoder tokenizer:`,
   `conditioning: text_encoder`, `seed=Random ...`,
   `lcm decoder: output ... shape=[1, 3, 256, 256]`,
   `preview: Smooth 256x256->256x256`, and `done: total=...`.
3. Generate a second time without changing the prompt. Confirm the logged random
   seed changes, the image is not identical, and the button reads `Reroll Seed`
   before the second tap.
4. Try short aliases such as `white mascot`, `cat logo`, `horse`, `astronaut`,
   `dog logo`, and `flying blue bird`. Confirm each logs
   `conditioning: text_encoder prompt="..." conditioningPrompt="..."` and still
   completes.
5. Try a previously unsupported prompt such as `spaceship banana`. Confirm it
   does not stop as unsupported, and instead logs `no preset match` followed by
   `using text encoder promptKey=spaceship_banana`,
   `conditioning: text_encoder prompt="spaceship banana"`, and
   `conditioningPrompt="spaceship banana, single subject, centered, full object visible, clean anime illustration, simple background"`.
   The logs should include `seed: mode=Random resolved=...`, and the run id
   should start with `spaceship_banana-`.
6. Record peak memory and typical memory while generating. The current 256px
   text-encoder-integrated baseline was reported around `140MB` peak.
