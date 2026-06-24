# watchOS Examples

This Xcode project contains all Watch-side tracks for Tiny Watch Image
Generator.

```sh
open TinyImageWatchApp.xcodeproj
```

## Schemes

| Scheme | Track | Use |
| --- | --- | --- |
| `TinyImageWatchApp` | Original MLP | Pure Swift demo with tracked `TinyWeights.bin`; no Core ML model download required. |
| `WatchStressTestApp` | Core ML Stress | Load/predict/memory probes for bundled `.mlmodelc` models. |
| `WatchTextEncoderSmokeApp` | Core ML Stress | Separated text encoder smoke cycle using `WATCH_TEXT_ENCODER_AUTORUN=1`. |
| `WatchPipelineSmokeApp` | Diffusion LCM256 | Current prompt-first LCM256 image-generation baseline. |

## TinyImageWatchApp

Use this when you want the simplest runnable app:

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Required tracked files:

- `TinyImageWatchApp/ContentView.swift`
- `TinyImageWatchApp/TinyImageWatchApp.swift`
- `TinyImageWatchApp/TinyWeights.bin`
- `../Sources/TinyWatchGenerator/TinyImageGenerator.swift`
- `../Sources/TinyWatchGenerator/TinyWeights.swift`

## WatchPipelineSmokeApp

Use this for the current adopted 256px diffusion baseline. It requires local
compiled Core ML bundles that are intentionally ignored by Git:

- `WatchPipelineSmokeApp/Models/lcm_unet_32x32_6bit_16p_part1.mlmodelc` ...
  `part16.mlmodelc`
- `WatchPipelineSmokeApp/Models/vae_decoder_256x256_noattn_4bit.mlmodelc`
- `WatchPipelineSmokeApp/TextEncoderAssets/clip_text_encoder_77.mlmodelc`
- `WatchPipelineSmokeApp/TextEncoderAssets/clip_vocab.json`
- `WatchPipelineSmokeApp/TextEncoderAssets/clip_merges.txt`

Build check:

```sh
xcodebuild -quiet \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme WatchPipelineSmokeApp \
  -destination generic/platform=watchOS \
  CODE_SIGNING_ALLOWED=NO \
  -derivedDataPath /private/tmp/watch_pipeline_check_build \
  build
```

Bundle verifier:

```sh
./.venv/bin/python tools/verify_watch_pipeline_smoke.py \
  --family lcm256 \
  --app /private/tmp/watch_pipeline_check_build/Build/Products/Debug-watchos/WatchPipelineSmokeApp.app
```

## Signing

Simulator builds do not require a development team. Physical Apple Watch runs
require your own Team in `Signing & Capabilities` and a unique bundle
identifier.
