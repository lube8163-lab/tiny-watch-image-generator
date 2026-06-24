# Mac LCM256 Full Quality Eval Summary

Run directory:

```text
reports/watch_lcm256_quality/full_lcm256_g6/
```

## Setup

- Prompt suite: `configs/watch_lcm256_quality_prompts.json`
- Images: 74 prompts x seeds `1, 7, 24, 42` = 296 images
- Failures: 0
- Guidance: 6
- Prompt expansion: enabled
- Preview: `Smooth`
- Compute: CPU-only Core ML on Mac
- Text encoder: `dist/lcm_dreamshaper_v7/text_encoder_probe/clip_text_encoder_77.mlpackage`
- UNet: `dist/lcm_dreamshaper_v7/unet_32x32_6bit.mlpackage`
- Decoder: `dist/lcm_dreamshaper_v7/vae_decoder_256x256_noattn_4bit.mlpackage`

## Result

296 images is enough for the current stage. It covers broad genre behavior,
seed variance, and obvious weak spots without turning evaluation into a large
dataset-management problem. Larger runs are useful only when comparing two model
variants that look close on this suite.

- Completed images: `296 / 296`
- Failures: `0`
- Mean generation time: `7.241s/image`
- Total measured generation time: `2143.440s` (`35.7` minutes)
- Output size on disk: about `44MB`

## Genre Summary

| Genre | Count | Mean Time | Flagged | Read |
| --- | ---: | ---: | ---: | --- |
| `nature` | 32 | `5.50s` | `6` | Strongest overall; mountains, ocean, beach, cabin, waterfall are stable. |
| `food` | 24 | `6.00s` | `7` | Very good for cake, ramen, sushi, hamburger; simple fruit/ice cream can look soft but still readable. |
| `fantasy` | 32 | `7.18s` | `10` | Strong for dragon, phoenix, crystal castle; weaker for unusual composites like flying whale. |
| `vehicles` | 24 | `8.49s` | `8` | Bus, train, airplane read well; sports car/sailboat depend more on seed. |
| `animals` | 40 | `6.48s` | `17` | Many are visually good despite soft flags; dolphin, tiger, owl, panda, blue bird are especially usable. |
| `objects` | 32 | `9.21s` | `14` | Teapot, boot, guitar, camera are good; bicycle, smartphone, umbrella are weaker. |
| `logos_icons` | 32 | `6.80s` | `16` | Good seeds exist for Apple logo, cat badge, paw print; abstract/logo text is unreliable. |
| `characters` | 32 | `6.72s` | `17` | Astronaut, robot, knight, chef can work; ballerina/superhero are weak. |
| `styles` | 24 | `6.93s` | `10` | Watercolor and sketch work; neon/origami are softer and more seed-dependent. |
| `relations` | 24 | `9.88s` | `9` | Better than expected for astronaut riding horse, cat spaceship, dog hat, bear bicycle; relation binding remains limited. |

## Prompt-Level Notes

Zero-flag prompts across all four seeds:

- `dolphin`
- `tiger`
- `knight`
- `dragon`
- `crystal castle`
- `phoenix bird`
- `hamburger`
- `ramen bowl`
- `strawberry cake`
- `sushi plate`
- `cat badge`
- `desert cactus`
- `forest cabin`
- `ocean wave`
- `snowy mountain`
- `sunset beach`
- `waterfall`
- `leather boot`
- `astronaut riding a horse`
- `bird holding flower`
- `sketch portrait`
- `watercolor flower`
- `steam train`
- `yellow bus`

Four-flag prompts across all four seeds:

- `ballerina`
- `green apple`
- `moonlit lake`
- `neon spiral`

The four-flag prompts are not all visually unusable. `green apple` and
`moonlit lake` can still look pleasant; the automatic flags mostly identify low
edge/contrast rather than semantic failure.

## Practical Read

The 256px baseline is good enough to adopt as the current Watch path. It is not
SDXL-like free prompt generation, but it handles short nouns and compact phrases
well enough that the image often reveals the intended prompt.

Best current use:

- short concrete noun prompts,
- simple scenes,
- animals,
- food,
- nature,
- simple objects,
- rerolling seeds until a good candidate appears.

Main weak spots:

- thin structures such as bicycles and fine limbs,
- pose-heavy human concepts,
- abstract style words,
- exact text/logo semantics,
- strict relation binding such as "holding" versus "near".

## Next Step

Use this 296-image suite as the normal quality gate:

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py \
  --out-dir reports/watch_lcm256_quality/full_lcm256_g6
```

When a future model/decoder/prompt-expansion change looks promising, run the
same suite into a new output directory and compare the genre contact sheets.
Reserve physical Apple Watch testing for memory peak, runtime, thermal behavior,
and final UI checks.
