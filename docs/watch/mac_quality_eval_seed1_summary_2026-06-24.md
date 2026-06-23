# Mac LCM256 Seed-1 Quality Eval Summary

Run directory:

```text
reports/watch_lcm256_quality/all_prompts_seed1_codex_g6/
```

## Setup

- Prompt suite: `configs/watch_lcm256_quality_prompts.json`
- Images: 74 prompts x seed `1`
- Failures: 0
- Guidance: 6
- Prompt expansion: enabled
- Compute: CPU-only Core ML on Mac
- Text encoder: `dist/lcm_dreamshaper_v7/text_encoder_probe/clip_text_encoder_77.mlpackage`
- UNet: `dist/lcm_dreamshaper_v7/unet_32x32_6bit.mlpackage`
- Decoder: `dist/lcm_dreamshaper_v7/vae_decoder_256x256_noattn_4bit.mlpackage`

## Timing

- Mean generation time: `10.607s/image`
- Total measured generation time: `784.889s` (`13.1` minutes)
- The full 296-image four-seed suite is expected to take roughly `55-90`
  minutes depending on cache state and machine load.

Relations and styles were the slowest groups in this run:

| Genre | Count | Mean Time |
| --- | ---: | ---: |
| `relations` | 6 | `25.17s` |
| `styles` | 6 | `21.99s` |
| `logos_icons` | 8 | `9.70s` |
| `fantasy` | 8 | `8.98s` |
| `nature` | 8 | `8.50s` |
| `animals` | 10 | `8.10s` |
| `vehicles` | 6 | `7.83s` |
| `characters` | 8 | `7.64s` |
| `food` | 6 | `7.47s` |
| `objects` | 8 | `6.34s` |

## Numeric Flags

The heuristic metrics flagged:

- `very_soft`: 28 images
- `low_contrast`: 4 images

Flagged counts by genre:

| Genre | Flagged / Count |
| --- | ---: |
| `logos_icons` | `6 / 8` |
| `characters` | `5 / 8` |
| `animals` | `3 / 10` |
| `fantasy` | `3 / 8` |
| `objects` | `3 / 8` |
| `relations` | `3 / 6` |
| `vehicles` | `3 / 6` |
| `styles` | `2 / 6` |
| `food` | `1 / 6` |
| `nature` | `1 / 8` |

## Visual Read

The strongest categories in the contact sheet were animals, nature scenes, food,
vehicles, and simple objects. Many of these are readable at a glance, and the
256px baseline looks meaningfully more useful than the earlier 128px and 192px
passes.

Logos/icons are often recognizable but numerically soft. This is not always a
bad result, because the model tends to create smooth badge-like marks, but this
genre should be reviewed visually rather than by the `very_soft` count alone.

Characters and relation prompts remain the main weak point. Single concepts such
as `robot`, `chef`, or `knight` are often plausible, while relation prompts such
as `cat in a spaceship` or `robot playing guitar` become more seed-dependent and
less compositionally reliable.

## Next Eval Step

Run the full four-seed suite when a larger sample is needed:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py \
  --out-dir reports/watch_lcm256_quality/full_lcm256_g6
```

Use the resulting contact sheets for semantic quality ranking, then reserve
Apple Watch testing for memory peak, runtime, thermal behavior, and final UX.
