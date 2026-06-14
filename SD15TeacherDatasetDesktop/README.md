# Stable Diffusion Core ML iPhone Test

This workspace is for building fresh Stable Diffusion -> Core ML test apps for iPhone, starting from model conversion rather than relying on Apple's published converted artifacts.

## Current direction

The current milestone is a reproducible conversion pipeline that is safer on 8 GB Macs and produces assets suitable for iPhone deployment:

- SDXL base 1.0 -> Core ML
- SD 1.5 -> Core ML
- iPhone/iPad-oriented 768x768 latent size (`96x96`)
- `SPLIT_EINSUM` attention for Neural Engine deployment
- sequential per-component conversion to reduce peak host RAM usage
- optional UNet chunking
- optional 6-bit quantization
- optional mixed-bit palettization pass for lower memory usage

## Why this setup

Apple's current public guidance for SDXL on iPhone/iPad is effectively:

- `iOS 17+`
- `CPU_AND_NE`
- `SPLIT_EINSUM`
- `reduceMemory = true`
- 768x768 output for mobile
- aggressive UNet compression for practical memory use on iPhone

The published `apple/coreml-stable-diffusion-xl-base-ios` benchmark also uses a heavily compressed UNet, so reproducing and controlling that pipeline ourselves is the right starting point if preconverted models were crashing on your devices.

## Files

- `scripts/convert_stable_diffusion.py`
  Stage-by-stage conversion wrapper around Apple's `python_coreml_stable_diffusion.torch2coreml` for both SDXL and SD 1.5.
- `scripts/convert_sdxl.py`
  Older SDXL-specific wrapper kept for reference.
- `scripts/apply_mixed_bit_palettization.py`
  Wrapper for applying a mixed-bit recipe to a converted UNet.
- `scripts/run_first_conversion.sh`
  Local convenience wrapper that uses the workspace venv and local Apple repo for the first SDXL base float16 conversion.
- `scripts/run_first_sd15_conversion.sh`
  Local convenience wrapper for a first SD 1.5 512x512 float16 conversion.
- `scripts/run_first_sd15_256_conversion.sh`
  Local convenience wrapper for a first SD 1.5 256x256 float16 conversion aimed at faster local generation.
- `scripts/run_first_pre_analysis.sh`
  Local convenience wrapper for Apple's mixed-bit recipe pre-analysis.
- `scripts/run_unet_6bit_palettization.sh`
  Applies 6-bit palettization to the converted UNet after float16 conversion succeeds.
- `scripts/run_sd15_unet_6bit_palettization.sh`
  Applies 6-bit palettization to the converted SD 1.5 UNet after float16 conversion succeeds.
- `scripts/convert_swinir_sr.py`
  Converts the official SwinIR lightweight x2 super-resolution weights into fixed-size Core ML upscalers for iPhone use. The app can reuse the `512x512` model as a tiled fallback for `768x768` SDXL outputs.
- `scripts/chunk_palettized_unet_for_ios.sh`
  Splits the palettized UNet into `UnetChunk1` and `UnetChunk2` for iPhone deployment.
- `scripts/verify_ios_resources.sh`
  Checks that the iPhone app has the expected Core ML resource set.
- `scripts/package_resources_for_ios.sh`
  Zips the finished `Resources/` directory for easier transfer/testing.
- `SDXLCoreMLTest.xcodeproj`
  Minimal SwiftUI iPhone test app wired to Apple's `StableDiffusion` runtime through a local package wrapper.
- `Vendor/StableDiffusionLocal`
  Local Swift package exposing only Apple's `StableDiffusion` library target so Xcode does not need the CLI dependency tree.

## Expected external dependencies

This repo does not vendor Apple's conversion code. The scripts expect a local clone of:

- `https://github.com/apple/ml-stable-diffusion`

Generated Core ML model packages and Hugging Face caches are intentionally not tracked in git. They are large local build artifacts and need to be regenerated with the scripts in this repo, or copied into `artifacts/` before making an App Store build.

Recommended local layout:

```text
SDXL_test/
├── README.md
├── artifacts/
├── recipes/
└── scripts/
```

Apple's repo can live anywhere, for example:

```text
~/src/ml-stable-diffusion
```

## Example flow

1. Create a Python environment inside Apple's repo and install its dependencies.
2. Run `scripts/convert_sdxl.py` to export SDXL base for iPhone-oriented settings.
3. If memory is still too high on device, apply 6-bit quantization or mixed-bit palettization to the UNet.
4. Feed the resulting `Resources/` directory into a Swift iOS app using Apple's `StableDiffusion` package.

For multi-model app testing, the recommended output layout is:

```text
artifacts/ios-models/
├── sd15/
│   ├── 512/Resources/
│   ├── 640/Resources/
│   └── 768/Resources/
└── sdxl/
    ├── 512/Resources/
    ├── 768/Resources/
    └── 1024/Resources/
```

Example conversion:

```bash
python3 scripts/convert_sdxl.py \
  --apple-repo ~/src/ml-stable-diffusion \
  --python-bin ./.venv/bin/python \
  --output-dir ./artifacts/sdxl-base-ios \
  --model-version stabilityai/stable-diffusion-xl-base-1.0 \
  --custom-vae-version madebyollin/sdxl-vae-fp16-fix \
  --skip-chunk-unet
```

Example fixed 6-bit UNet palettization after conversion:

```bash
python3 scripts/apply_fixed_bit_palettization.py \
  --mlpackage-path ./artifacts/sdxl-base-ios/Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet.mlpackage \
  --output-mlpackage-path ./artifacts/sdxl-base-ios/Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet_6bit.mlpackage \
  --nbits 6 \
  --compile-to ./artifacts/sdxl-base-ios/Resources \
  --final-name Unet
```

Example chunking the palettized UNet for iPhone runtime:

```bash
./scripts/chunk_palettized_unet_for_ios.sh
```

Example mixed-bit pre-analysis:

```bash
python3 scripts/run_mixed_bit_pre_analysis.py \
  --apple-repo ~/src/ml-stable-diffusion \
  --python-bin ./.venv/bin/python \
  --output-dir ./recipes \
  --model-version stabilityai/stable-diffusion-xl-base-1.0
```

Example mixed-bit application:

```bash
python3 scripts/apply_mixed_bit_palettization.py \
  --apple-repo ~/src/ml-stable-diffusion \
  --python-bin ./.venv/bin/python \
  --converted-model-dir ./artifacts/sdxl-base-ios \
  --pre-analysis-json ./recipes/stabilityai_stable-diffusion-xl-base-1.0.json \
  --selected-recipe recipe_4.04_bit_mixedpalette
```

## Notes

- The conversion wrappers default to separate subprocesses per model component because Apple explicitly recommends this for low-memory conversion on 8 GB systems.
- The conversion and pre-analysis wrappers default to a workspace-local cache at `.cache/huggingface`, avoiding permission issues with `~/.cache`.
- The recommended path in this workspace is float16 conversion first, then separate UNet-only palettization, then chunking that palettized UNet for iPhone runtime.
- Apple `torch2coreml --quantize-nbits` also tries to quantize existing text encoders, which failed here.
- `madebyollin/sdxl-vae-fp16-fix` is exposed as an option because Apple's own SDXL mobile benchmark used it to keep the VAE on float16.
- The scripts currently target the base model only. Refiner support can be added after the iPhone baseline is stable.

## Next step

After the conversion pipeline is validated locally, the next implementation step is a minimal iOS app that:

- downloads or locates `Resources/`
- loads `StableDiffusionPipeline` with `reduceMemory = true`
- uses `.cpuAndNeuralEngine`
- runs a fixed prompt/seed smoke test on device

## iPhone test app

The workspace now includes a minimal app at `SDXLCoreMLTest.xcodeproj`.

What it does:

- uses both `StableDiffusionXLPipeline` and `StableDiffusionPipeline`
- forces `reduceMemory = true`
- uses `.cpuAndNeuralEngine`
- targets `iOS 17+`
- supports model selection between `SDXL` and `SD 1.5`
- supports resolution presets and switches between preconverted resource folders
- preloads resources on launch and releases them on backgrounding

Expected resource location inside the app sandbox:

```text
Documents/Models/<model>/<resolution>/Resources
```

Examples:

```text
Documents/Models/sdxl/768/Resources
Documents/Models/sd15/512/Resources
```

Recommended test flow:

1. Build and install `SDXLCoreMLTest` on the iPhone.
2. Copy the converted `Resources/` directory into the app's Documents folder using Finder file sharing.
3. Launch the app and choose the model family and resolution that match the copied resources.
4. Run the built-in prompt first before changing settings.

## Mac desktop target

The Mac Catalyst desktop prototype lives in a separate target and scheme: `SDXLDesktopCoreMLTest`. The original `SDXLCoreMLTest` target remains the iPhone-oriented SDXL build.

`SDXLDesktopCoreMLTest` is oriented around `SD 1.5` first and uses bundled `512x512` resources by default.

`256x256` resources can be converted with `scripts/run_first_sd15_256_conversion.sh`, but native `256x256` SD 1.5 Core ML output currently produces tiled/color-corrupted images in this runtime. Keep those resources for experimentation only until that conversion/runtime issue is resolved.

Bundled desktop resources are expected under:

```text
BundledResources/sd15/256/Resources
BundledResources/sd15/512/Resources
```

## First commands in this workspace

After accepting the SDXL license on Hugging Face and making a token available, the shortest path is:

```bash
chmod +x scripts/run_first_conversion.sh scripts/run_first_pre_analysis.sh scripts/run_unet_6bit_palettization.sh scripts/chunk_palettized_unet_for_ios.sh
HF_TOKEN=... ./scripts/run_first_conversion.sh
```

For a first SD 1.5 build:

```bash
HF_TOKEN=... ./scripts/run_first_sd15_conversion.sh
```

Then compress only the SD 1.5 UNet:

```bash
./scripts/run_sd15_unet_6bit_palettization.sh
```

Then compress only the UNet:

```bash
./scripts/run_unet_6bit_palettization.sh
```

Then split that palettized UNet into two runtime chunks:

```bash
./scripts/chunk_palettized_unet_for_ios.sh
```

Verify the final mobile resource set:

```bash
chmod +x scripts/*.sh
./scripts/verify_ios_resources.sh
```

Optional zip packaging:

```bash
./scripts/package_resources_for_ios.sh
```

Optional mixed-bit recipe generation:

```bash
HF_TOKEN=... ./scripts/run_first_pre_analysis.sh
```

Optional super-resolution conversion:

```bash
python3 scripts/convert_swinir_sr.py --sizes 512 --compile
```

## Store support pages

GitHub Pages can serve the App Store support and privacy policy pages from `docs/`:

- `docs/support.html`
- `docs/privacy.html`

After enabling Pages for the repository, use the public `support.html` URL as the App Store Connect Support URL and the public `privacy.html` URL as the Privacy Policy URL.

Unsigned compile-only verification command used here:

```bash
xcodebuild build \
  -project SDXLCoreMLTest.xcodeproj \
  -scheme SDXLCoreMLTest \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  -derivedDataPath .build/xcode-derived-data-nosign \
  -clonedSourcePackagesDirPath .build/xcode-source-packages
```

## Sources

- [Apple ml-stable-diffusion README](https://github.com/apple/ml-stable-diffusion)
- [Hugging Face: Stable Diffusion XL on Mac with Advanced Core ML Quantization](https://huggingface.co/blog/stable-diffusion-xl-coreml)
