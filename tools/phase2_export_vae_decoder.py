#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from research_common import (
    ROOT,
    directory_size,
    pick_torch_device,
    require_diffusion_stack,
    resolve_model_path,
    select_candidate,
    torch_dtype_for_device,
    write_manifest,
)


def require_coremltools():
    try:
        import coremltools as ct
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "coremltools and numpy are required:\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        ) from exc
    return ct, np


def target_from_name(ct, name: str):
    try:
        return getattr(ct.target, name)
    except AttributeError as exc:
        known = [item for item in dir(ct.target) if item.startswith("iOS") or item.startswith("macOS")]
        raise SystemExit(f"unknown Core ML deployment target '{name}'. Known examples: {', '.join(known)}") from exc


def make_wrapper(torch, vae):
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.vae = wrapped
            self.scaling_factor = float(getattr(wrapped.config, "scaling_factor", 0.18215))

        def forward(self, latents):
            decoded = self.vae.decode(latents / self.scaling_factor, return_dict=False)[0]
            return decoded

    return VAEDecoderWrapper(vae).eval()


def replace_mid_attention_with_identity(torch, vae) -> bool:
    mid_block = getattr(getattr(vae, "decoder", None), "mid_block", None)
    attentions = getattr(mid_block, "attentions", None)
    if attentions is None or len(attentions) == 0:
        return False

    class IdentityAttention(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    for index in range(len(attentions)):
        attentions[index] = IdentityAttention()
    return True


def pick_export_device(torch, name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        return torch.device("mps")
    return pick_torch_device(torch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: export only the VAE decoder as Core ML.")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-height", type=int, default=64)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--latent-scale", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--deployment-target", default="iOS16")
    parser.add_argument("--compute-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--torch-precision", choices=["fp16", "fp32", "auto"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--drop-mid-attention", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out = Path(args.out) if args.out else ROOT / "dist" / key / f"vae_decoder_{args.output_width}x{args.output_height}.mlpackage"
    manifest = Path(args.manifest) if args.manifest else out.with_suffix(".json")

    if args.output_height % args.latent_scale or args.output_width % args.latent_scale:
        raise SystemExit("output height and width must be divisible by latent scale")

    torch, diffusers = require_diffusion_stack()
    ct, np = require_coremltools()
    device = pick_export_device(torch, args.device)
    if args.torch_precision == "fp32":
        dtype = torch.float32
    elif args.torch_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch_dtype_for_device(torch, device)

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

    if not hasattr(pipe, "vae") or pipe.vae is None:
        raise SystemExit(f"{model_id} does not expose a VAE component")

    dropped_mid_attention = False
    if args.drop_mid_attention:
        dropped_mid_attention = replace_mid_attention_with_identity(torch, pipe.vae)

    wrapper = make_wrapper(torch, pipe.vae).to(device=device, dtype=dtype)
    latent_h = args.output_height // args.latent_scale
    latent_w = args.output_width // args.latent_scale
    latent_channels = int(getattr(pipe.vae.config, "latent_channels", 4))
    sample = torch.randn(1, latent_channels, latent_h, latent_w, dtype=dtype, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, sample)
    trace_elapsed = time.perf_counter() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    ml_dtype = np.float16 if args.compute_precision == "fp16" else np.float32
    convert_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="latents", shape=sample.shape, dtype=ml_dtype)],
        outputs=[ct.TensorType(name="decoded")],
        minimum_deployment_target=target_from_name(ct, args.deployment_target),
        compute_precision=ct.precision.FLOAT16 if args.compute_precision == "fp16" else ct.precision.FLOAT32,
    )
    mlmodel.save(out)
    convert_elapsed = time.perf_counter() - convert_start

    write_manifest(
        manifest,
        {
            "phase": "phase2_export_vae_decoder",
            "candidate": key,
            "model": model_id,
            "source_repo": candidate["repo"],
            "output": str(out),
            "output_bytes": directory_size(out),
            "latent_shape": list(sample.shape),
            "decoded_shape": [1, 3, args.output_height, args.output_width],
            "device": device.type,
            "torch_dtype": str(dtype),
            "compute_precision": args.compute_precision,
            "deployment_target": args.deployment_target,
            "dropped_mid_attention": dropped_mid_attention,
            "trace_elapsed_seconds": trace_elapsed,
            "convert_elapsed_seconds": convert_elapsed,
        },
    )
    print(out)
    print(manifest)


if __name__ == "__main__":
    main()
