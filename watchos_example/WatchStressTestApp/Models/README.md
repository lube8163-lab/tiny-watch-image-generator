# WatchStressTestApp Models

Drop compiled `.mlmodelc` directories here to bundle them into the stress test app.

Example:

```sh
xcrun coremlcompiler compile \
  dist/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.mlpackage \
  watchos_example/WatchStressTestApp/Models
```

The app scans the bundle recursively for `.mlmodelc` directories and logs load/prediction results to both the on-screen log and the Xcode console with the `[WatchStress]` prefix.

## Text Encoder Smoke

The current text encoder probe bundles:

- `clip_text_encoder_77.mlmodelc`
- `input_ids_i32.bin`
- `reference_hidden_states_f16.bin`
- `text_encoder_probe_prompts.json`

The model is exported from `models/lcm_dreamshaper_v7/text_encoder` with fixed
input shape `1x77` and output shape `1x77x768`. The bundled probe asset
currently includes `cat mascot`, `horse`, `astronaut`, `dog logo`,
`flying blue bird`, and `spaceship banana`.

For physical-device checks, prefer the `WatchTextEncoderSmokeApp` scheme. It
launches this target with `WATCH_TEXT_ENCODER_AUTORUN=1`, loads the text encoder
as a transient model, predicts all bundled prompts, compares against
`reference_hidden_states_f16.bin`, releases the model scope, purges the Core ML
cache, and leaves the app ready for a later generation load.

Regenerate the Core ML package and assets with:

```sh
./.venv/bin/python tools/export_clip_text_encoder_coreml.py \
  --candidate lcm_dreamshaper_v7 \
  --local-files-only \
  --prompts \
    "cat mascot" \
    "horse" \
    "astronaut" \
    "dog logo" \
    "flying blue bird" \
    "spaceship banana" \
  --convert
```

Then compile and copy the model with:

```sh
rm -rf /private/tmp/text_encoder_probe_compile
mkdir -p /private/tmp/text_encoder_probe_compile
xcrun coremlcompiler compile \
  dist/lcm_dreamshaper_v7/text_encoder_probe/clip_text_encoder_77.mlpackage \
  /private/tmp/text_encoder_probe_compile
rm -rf watchos_example/WatchStressTestApp/Models/clip_text_encoder_77.mlmodelc
cp -R /private/tmp/text_encoder_probe_compile/clip_text_encoder_77.mlmodelc \
  watchos_example/WatchStressTestApp/Models/
cp dist/lcm_dreamshaper_v7/text_encoder_probe/input_ids_i32.bin \
  watchos_example/WatchStressTestApp/Models/
cp dist/lcm_dreamshaper_v7/text_encoder_probe/reference_hidden_states_f16.bin \
  watchos_example/WatchStressTestApp/Models/
cp dist/lcm_dreamshaper_v7/text_encoder_probe/text_encoder_probe_prompts.json \
  watchos_example/WatchStressTestApp/Models/
```
