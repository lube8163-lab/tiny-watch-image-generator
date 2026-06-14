#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from phase2_export_vae_decoder import require_coremltools, target_from_name
from phase3_export_unet import patch_coremltools_int_cast, set_attention_processor
from research_common import ROOT, directory_size, require_diffusion_stack, resolve_model_path, select_candidate, write_manifest


def make_wrapper(torch, unet):
    class SDUNetWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.unet = wrapped

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

    return SDUNetWrapper(unet).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fixed-shape Stable Diffusion UNet as Core ML.")
    parser.add_argument("--candidate", default="segmind_tiny_sd")
    parser.add_argument("--model", default=None)
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=77)
    parser.add_argument("--timestep", type=float, default=999.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--deployment-target", default="iOS16")
    parser.add_argument("--compute-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--torch-precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--attention-processor", choices=["default", "eager"], default="eager")
    parser.add_argument("--patch-coreml-int-cast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out = Path(args.out) if args.out else ROOT / "dist" / key / f"unet_sd_{args.latent_width}x{args.latent_height}.mlpackage"
    manifest = Path(args.manifest) if args.manifest else out.with_suffix(".json")

    torch, diffusers = require_diffusion_stack()
    ct, np = require_coremltools()
    dtype = torch.float16 if args.torch_precision == "fp16" else torch.float32
    device = torch.device("cpu")

    pipeline_cls = getattr(diffusers, candidate["pipeline"], None)
    if pipeline_cls is None:
        raise SystemExit(f"diffusers does not expose {candidate['pipeline']}; update diffusers.")

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet.to(device=device, dtype=dtype).eval()
    set_attention_processor(unet, args.attention_processor)
    wrapper = make_wrapper(torch, unet)

    in_channels = int(unet.config.in_channels)
    cross_attention_dim = int(unet.config.cross_attention_dim)
    sample = torch.randn(1, in_channels, args.latent_height, args.latent_width, dtype=dtype, device=device)
    timestep = torch.tensor([args.timestep], dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, args.prompt_length, cross_attention_dim, dtype=dtype, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (sample, timestep, encoder_hidden_states))
    trace_elapsed = time.perf_counter() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.patch_coreml_int_cast:
        patch_coremltools_int_cast()
    ml_dtype = np.float16 if args.compute_precision == "fp16" else np.float32
    convert_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape, dtype=ml_dtype),
            ct.TensorType(name="timestep", shape=timestep.shape, dtype=ml_dtype),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape, dtype=ml_dtype),
        ],
        outputs=[ct.TensorType(name="noise_pred")],
        minimum_deployment_target=target_from_name(ct, args.deployment_target),
        compute_precision=ct.precision.FLOAT16 if args.compute_precision == "fp16" else ct.precision.FLOAT32,
    )
    mlmodel.save(out)
    convert_elapsed = time.perf_counter() - convert_start

    write_manifest(
        manifest,
        {
            "phase": "phase3_export_sd_unet",
            "candidate": key,
            "model": model_id,
            "source_repo": candidate["repo"],
            "output": str(out),
            "output_bytes": directory_size(out),
            "sample_shape": list(sample.shape),
            "timestep_shape": list(timestep.shape),
            "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
            "torch_dtype": str(dtype),
            "compute_precision": args.compute_precision,
            "deployment_target": args.deployment_target,
            "attention_processor": args.attention_processor,
            "patched_coreml_int_cast": args.patch_coreml_int_cast,
            "trace_elapsed_seconds": trace_elapsed,
            "convert_elapsed_seconds": convert_elapsed,
        },
    )
    print(out)
    print(manifest)


if __name__ == "__main__":
    main()
