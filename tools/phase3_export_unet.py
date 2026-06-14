#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from research_common import (
    ROOT,
    directory_size,
    require_diffusion_stack,
    resolve_model_path,
    select_candidate,
    write_manifest,
)
from phase2_export_vae_decoder import require_coremltools, target_from_name


def make_wrapper(torch, unet):
    class UNetWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.unet = wrapped

        def forward(self, sample, timestep, encoder_hidden_states, timestep_cond):
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                return_dict=False,
            )[0]

    return UNetWrapper(unet).eval()


def make_guidance_embedding(torch, w, embedding_dim: int, dtype):
    # Matches Diffusers LatentConsistencyModelPipeline.get_guidance_scale_embedding.
    w = w * 1000.0
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, dtype=dtype)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def set_attention_processor(unet, mode: str) -> None:
    if mode == "default":
        return
    if mode == "eager":
        from diffusers.models.attention_processor import AttnProcessor

        unet.set_attn_processor(AttnProcessor())
        return
    raise SystemExit(f"unknown attention processor mode: {mode}")


def patch_coremltools_int_cast() -> None:
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    def patched_int(context, node):
        inputs = ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            value = x.val
            if isinstance(value, np.ndarray):
                value = value.item()
            res = mb.const(val=int(value), name=node.name)
        elif len(x.shape) > 0:
            squeezed = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=squeezed, dtype="int32", name=node.name)
        else:
            res = mb.cast(x=x, dtype="int32", name=node.name)
        context.add(res, node.name)

    ops._TORCH_OPS_REGISTRY.set_func_by_name(patched_int, "int")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: export a fixed-shape LCM/SD UNet as Core ML.")
    parser.add_argument("--candidate", default="lcm_dreamshaper_v7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=77)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
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
    out = Path(args.out) if args.out else ROOT / "dist" / key / f"unet_{args.latent_width}x{args.latent_height}.mlpackage"
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
    time_cond_dim = int(getattr(unet.config, "time_cond_proj_dim", 0) or 0)
    if time_cond_dim == 0:
        raise SystemExit("UNet does not expose time_cond_proj_dim; add a separate SD UNet export path.")

    sample = torch.randn(1, in_channels, args.latent_height, args.latent_width, dtype=dtype, device=device)
    timestep = torch.tensor([args.timestep], dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, args.prompt_length, cross_attention_dim, dtype=dtype, device=device)
    timestep_cond = make_guidance_embedding(
        torch,
        torch.tensor([args.guidance_scale - 1.0], dtype=dtype, device=device),
        time_cond_dim,
        dtype,
    )

    start = time.perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (sample, timestep, encoder_hidden_states, timestep_cond))
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
            ct.TensorType(name="timestep_cond", shape=timestep_cond.shape, dtype=ml_dtype),
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
            "phase": "phase3_export_unet",
            "candidate": key,
            "model": model_id,
            "source_repo": candidate["repo"],
            "output": str(out),
            "output_bytes": directory_size(out),
            "sample_shape": list(sample.shape),
            "timestep_shape": list(timestep.shape),
            "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
            "timestep_cond_shape": list(timestep_cond.shape),
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
