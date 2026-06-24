# Watch Text Encoder Smoke

Updated: 2026-06-24

This is the separated physical-device probe that was used before wiring free
prompt input into `WatchPipelineSmokeApp`. The text encoder is now integrated
into the LCM256 baseline, but this scheme remains useful when the encoder needs
to be checked in isolation. Use the `WatchTextEncoderSmokeApp` scheme for the
single-purpose text encoder run. It launches the existing stress-test target
with `WATCH_TEXT_ENCODER_AUTORUN=1`, so the app opens as a text-encoder smoke
screen and runs the separated cycle automatically.

## Why This Exists

Earlier prompt-first builds resolved typed text to bundled preset embeddings.
The separated text encoder probe confirmed that the Watch could encode the
user's exact short prompt before the generator loaded. That successful direction
is now part of `WatchPipelineSmokeApp`; keep this note as the diagnostic recipe
for rerunning the encoder alone.

## Current Probe

The probe uses the local LCM DreamShaper CLIP text encoder:

- Source: `models/lcm_dreamshaper_v7/text_encoder`
- Input: `input_ids`, fixed shape `1x77`, `Int32`
- Output: `hidden_states`, fixed shape `1x77x768`
- Core ML package: `dist/lcm_dreamshaper_v7/text_encoder_probe/clip_text_encoder_77.mlpackage`
- Bundled compiled model: `watchos_example/WatchStressTestApp/Models/clip_text_encoder_77.mlmodelc`
- Bundled reference: `watchos_example/WatchStressTestApp/Models/reference_hidden_states_f16.bin`
- Recent model size: about `235M`
- Recent `WatchStressTestApp.app` bundle size: about `416M`

The bundled probe currently contains six prompts:

- `cat mascot`
- `horse`
- `astronaut`
- `dog logo`
- `flying blue bird`
- `spaceship banana`

The stress app loads the text encoder once, predicts all six prompts by swapping
the `input_ids` row, compares each output against the matching row in
`reference_hidden_states_f16.bin`, then releases the model and purges the Core
ML cache.

## Build

```sh
xcodebuild -quiet \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme WatchTextEncoderSmokeApp \
  -destination generic/platform=watchOS \
  CODE_SIGNING_ALLOWED=NO \
  -derivedDataPath /private/tmp/watch_text_encoder_stress_build \
  build
```

## Device Checklist

Run these on a physical Apple Watch:

1. In Xcode, select the `WatchTextEncoderSmokeApp` scheme and run it on the
   physical Watch.
2. Confirm the app title is `Text Encoder`.
3. The smoke cycle should start automatically. If it does not, tap
   `Run Text Encoder`.
4. Confirm the app stays alive and capture peak memory, typical memory, load
   time, predict time, and post-release memory from Xcode/device tools.
5. Confirm the console includes `text encoder prompts: count=6`, all six
   `text encoder prompt: n/6 ...` lines, `input_ids: loaded 77 ids prompt=n`,
   output `hidden_states`, per-prompt `predict:` timing lines,
   `text encoder compare: prompt="..." count=59136 ...`, and
   `text encoder separated: ready for generation load`.

Do not use the generic `Load & Retain` button for this check. That button is
hidden in the `WatchTextEncoderSmokeApp` scheme, but remains available in the
plain `WatchStressTestApp` scheme.

The product direction is the separated cycle: load the CLIP text encoder,
create/check the embedding, release it, purge the Core ML cache, then start the
generation load.

## Logs To Save

Useful `[WatchStress]` lines:

```text
scan: found ...
model: clip_text_encoder_77.mlmodelc
start: text encoder separated cycle
text encoder separated: begin transient load/predict/release
text encoder prompts: count=6
load: clip_text_encoder_77.mlmodelc ...
desc: inputs=[input_ids] outputs=[hidden_states]
text encoder prompt: 1/6 "cat mascot"
input_ids: loaded 77 ids prompt=1 from input_ids_i32.bin
input: input_ids multiArray shape=1x77 ...
predict: clip_text_encoder_77.mlmodelc #1 ... outputs=[hidden_states]
text encoder compare: prompt="cat mascot" count=59136 rms=... max=...
text encoder prompt: 2/6 "horse"
...
text encoder compare: prompt="spaceship banana" count=59136 rms=... max=...
text encoder separated: transient model scope ended
cache: ...
text encoder separated: ready for generation load
```

## Interpretation

If the FP16 text encoder fails to load, predicts unreliably, or causes large
memory spikes, do not integrate it into the generation app. The next direction
should be a smaller text encoder, a distilled prompt encoder, or a compact
learned embedding table for short watch prompts.

If it loads, predicts, releases cleanly, and the comparison error is small, the
next local step is to add an optional separated text-encoder path to
`WatchPipelineSmokeApp` while keeping preset embeddings as regression tests.
