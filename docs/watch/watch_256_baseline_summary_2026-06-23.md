# Watch 256px Baseline Summary

Updated: 2026-06-23

## Adopted Baseline

The adopted Apple Watch text-to-image baseline is:

- Scheme: `WatchPipelineSmokeApp`
- Pipeline: `LCM256 6b`
- Text conditioning: transient on-device CLIP text encoder
- UNet: 16 streamed 6-bit chunks, `lcm_unet_32x32_6bit_16p_part1...16`
- Decoder: `vae_decoder_256x256_noattn_4bit`
- Scheduler latent shape: `1x4x32x32`
- Decoder output shape: `1x3x256x256`
- Guidance: `6`
- Seed mode: `Random`, with reroll as the normal exploration path
- Preview: direct `Smooth` 256px display
- Compute units: CPU-only

## Device Evidence

The 256px path completed on physical Apple Watch with observed peak memory
around `140MB`.

Representative Xcode log timings after switching the default preview from
`Sharp x2` to direct `Smooth`:

| Prompt | Total | Text Encoder | UNet Step 1 | Later Steps | Decoder | Preview |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flying blue bird` | `59.655s` | `3.443s` | `31.604s` | `6.283-6.475s` | `4.430s` | `0.000s + 0.001s` |
| `smartphone` | `58.248s` | `2.731s` | `30.838s` | `6.239-8.064s` | `3.590s` | `0.000s + 0.000s` |
| `Apple logo` | `62.721s` | `2.832s` | `32.311s` | `6.571-10.271s` | `3.558s` | `0.000s + 0.000s` |

The first step is dominated by initial chunk loading. Later steps mostly reuse
the cache and are much faster. Direct `Smooth` preview removes the expensive
512px `Sharp x2` postprocess path.

## Quality Position

256px is now the best practical resolution tested on device. It improves over
128px and 192px, while keeping the same streamed UNet weight footprint and
remaining within the observed memory envelope.

Further resolution increases may be technically possible, but the expected
quality gain is unlikely to justify the added runtime, decoder cost, activation
memory, and preview cost. A major quality jump likely requires model training,
distillation, or decoder replacement rather than another resolution bump.

See also:

- `pipeline_quality_notes.md`
- `mac_quality_eval.md`
- `future_quality_breakthroughs.md`

## Mac Evaluation Policy

Model-quality evaluation can be run on Mac as long as it uses the same assets
and runtime assumptions:

- same prompt normalization and prompt expansion,
- same tokenizer and text encoder,
- same scheduler JSON,
- same compressed Core ML UNet chunks,
- same decoder,
- same random seed,
- `MLComputeUnits.cpuOnly`,
- same RGB conversion and preview mode.

Mac results should be treated as quality-equivalent for ranking prompts, seeds,
guidance settings, and candidate model variants. They are not guaranteed to be
bit-for-bit identical to watchOS because Core ML compilation and low-level CPU
kernels can differ between macOS and watchOS, but the difference should be small
enough for contact sheets and candidate selection.

Final runtime, memory, thermal behavior, and any borderline candidate should
still be verified on the physical Apple Watch.

## Recommended Split

- Use Mac for broad quality sweeps, seed search, prompt suites, contact sheets,
  and comparing model variants.
- Use Apple Watch for runtime, memory peak, thermal behavior, app UX, and final
  go/no-go checks.
- Keep ANE out of the quality path unless a future branch explicitly probes
  accelerator behavior. The current product path is CPU-only.
