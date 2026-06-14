# Watch Eval Baseline

Generated locally from the current Swift watch generator.

```sh
python3 tools/make_watch_eval_contact_sheet.py \
  --out-dir reports/watch_eval/baseline_current \
  --seeds 0
```

The generated contact sheets live under `reports/watch_eval/baseline_current/`. This directory is intentionally ignored by Git.

## Current Read

The release evaluator is fast enough for local iteration:

- 128x128 generation on this Mac: roughly 0.45-0.61 seconds per image
- Default baseline subset: 24 images in roughly 13 seconds
- Debug SwiftPM builds are much slower and should not be used for normal eval sheets

Qualitative read from `contact_sheet_all.png`:

- Core subjects such as `cat`, `dog`, `car`, and `flower` are recognizable, but still soft and partly blended into the background.
- Newer v6 subjects such as `astronaut`, `alien`, `dragon`, and `penguin` are surprisingly readable.
- Color prompts affect the image, but color is not consistently bound to the intended subject.
- Action prompts such as `running`, `sitting`, `flying`, and `swimming` are weak. They mostly change texture/pose noise rather than reliable geometry.
- Style prompts such as `anime`, `watercolor`, `sketch`, and `photo` are weakly expressed.
- Japanese aliases route to the right general concepts, but inherit the same color/action weakness.
- Background noise remains a major visual issue, even after lightweight watch-side postprocess.

## Next Training Target

The next cloud dataset should focus less on adding many new nouns and more on slot-conditioned examples:

- Repeated clean single-subject examples for existing eval nouns.
- Color binding pairs such as `red car`, `blue bird`, `white cat`, `black dog`, `pink flower`.
- Pose/action pairs with strong silhouettes such as `running dog`, `sitting cat`, `flying bird`, `swimming fish`.
- View pairs such as `side view car`, `top view pizza`, `front view clock`.
- Style pairs with deliberately distinct teacher images: `icon`, `cartoon`, `sketch`, `toy`, and `watercolor`.
- Plain matte background, low texture, no repeated objects, no scenery.

## When RunPod Is Needed

RunPod is not needed for local eval and prompt/config preparation. It becomes useful when we are ready to generate a v8 teacher supplement and retrain:

1. Generate eval-aligned SDXL teacher images.
2. Filter/validate them into a compact quality-diverse dataset.
3. Retrain `TinyWeights.bin`.
4. Re-run the same watch eval sheet and compare before/after.

Until then, local work should focus on tightening eval coverage and preparing the teacher prompt config.
