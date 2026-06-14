#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from research_common import ROOT, pick_torch_device, require_diffusion_stack, resolve_model_path, select_candidate, torch_dtype_for_device, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final diffusion latents as raw float16 for Watch decoder-only tests.")
    parser.add_argument("--candidate", default="lcm_dreamshaper_v7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="a tiny watercolor landscape, crisp details")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out = Path(args.out) if args.out else ROOT / "reports" / "phase2" / key / f"seed{args.seed}_{args.width}x{args.height}.latents.f16"
    preview = Path(args.preview) if args.preview else out.with_suffix(".png")

    torch, diffusers = require_diffusion_stack()
    device = pick_torch_device(torch)
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

    generator_device = "cpu" if device.type == "mps" else device.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed)

    with torch.inference_mode():
        latent_result = pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="latent",
        )
        latents_device = latent_result.images.detach()
        decoded = pipe.vae.decode(latents_device / pipe.vae.config.scaling_factor, return_dict=False)[0]
        preview_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        latents = latents_device.to("cpu", dtype=torch.float16).contiguous()

    out.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(latents.numpy().tobytes())
    preview_image.save(preview)

    manifest = out.with_suffix(".json")
    write_manifest(
        manifest,
        {
            "phase": "phase2_export_latents",
            "candidate": key,
            "model": model_id,
            "source_repo": candidate["repo"],
            "prompt": args.prompt,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "latent_shape": list(latents.shape),
            "latent_dtype": "float16",
            "latent_bytes": out.stat().st_size,
            "latent_file": str(out),
            "preview": str(preview),
        },
    )
    print(out)
    print(preview)
    print(manifest)


if __name__ == "__main__":
    main()
