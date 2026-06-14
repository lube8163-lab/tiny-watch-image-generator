# iOS device build

`ios_example/TinyImageIOSApp.xcodeproj` is a SwiftUI app for checking the current generator on iPhone.

The app now includes the DPMSolver/Core ML 128x128 path from `dist/segmind_tiny_sd`:

- `unet_sd_16x16_6bit.mlmodelc`
- `vae_decoder_128x128_noattn_4bit.mlmodelc`
- `vae_decoder_128x128_noattn.mlmodelc`
- 24 preset prompt embeddings, unconditional embedding, and scheduler assets

The bundled resources are about 351 MB. Text input is mapped to the closest exported preset because the full text encoder is not bundled in this iOS target yet.

The UNet output on iPhone is padded (`MLMultiArray.strides` can differ from the logical shape), so the app reads Core ML outputs through shape/stride-aware indexing instead of assuming contiguous memory.

This route uses classifier-free guidance and a 30-step DPMSolverMultistep scheduler. The app exposes guidance strength and VAE decoder mode so device results can be compared before deciding whether model training, a different distilled checkpoint, or a larger text-conditioning path is needed.

## Quality checks

Use the same preset and seed while changing one setting at a time:

1. Compare `VAE 4-bit` and `VAE FP16`.
2. Try guidance values `4`, `6`, `8`, `10`, and `12`.
3. Check Xcode console lines for `finalLatents`, `decoded`, and `image clippedChannels`.

If `VAE FP16` reduces clipping or improves colors but prompt matching remains weak, the decoder quantization is only part of the bottleneck. If guidance changes saturation but not structure, the likely bottleneck is the model or text-conditioning quality rather than the iOS app runtime.

## Run on iPhone

1. Open `ios_example/TinyImageIOSApp.xcodeproj` in Xcode.
2. Select the `TinyImageIOSApp` target.
3. In `Signing & Capabilities`, choose your development team.
4. Select a connected iPhone and run.

The app compiles the existing `Sources/TinyWatchGenerator` Swift files directly into the iOS app target. The generator logic and embedded weights are shared with the Swift package source.

Command-line simulator build:

```sh
xcodebuild \
  -project ios_example/TinyImageIOSApp.xcodeproj \
  -scheme TinyImageIOSApp \
  -destination 'generic/platform=iOS Simulator' \
  build
```

Command-line device compile check without signing:

```sh
xcodebuild \
  -project ios_example/TinyImageIOSApp.xcodeproj \
  -scheme TinyImageIOSApp \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```
