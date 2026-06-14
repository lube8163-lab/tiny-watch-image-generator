#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from research_common import (
    ROOT,
    pick_torch_device,
    require_diffusion_stack,
    resolve_model_path,
    select_candidate,
    torch_dtype_for_device,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: generate images with a candidate Diffusers model on Mac.")
    parser.add_argument("--candidate", default=None, help="Candidate key from configs/model_candidates.json")
    parser.add_argument("--model", default=None, help="Override Hugging Face repo id or local model path")
    parser.add_argument("--prompt", default="a tiny watercolor landscape, crisp details")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-safety-checker", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    width = args.width or candidate["initial_width"]
    height = args.height or candidate["initial_height"]
    steps = args.steps or candidate["initial_steps"]
    out = Path(args.out) if args.out else ROOT / "reports" / "phase1" / key / f"seed{args.seed}_{width}x{height}.png"

    torch, diffusers = require_diffusion_stack()
    device = pick_torch_device(torch)
    dtype = torch_dtype_for_device(torch, device)

    pipeline_cls = getattr(diffusers, candidate["pipeline"], None)
    if pipeline_cls is None:
        raise SystemExit(f"diffusers does not expose {candidate['pipeline']}; update diffusers.")

    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }
    pipe = pipeline_cls.from_pretrained(model_id, **load_kwargs)
    pipe = pipe.to(device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    generator_device = "cpu" if device.type == "mps" else device.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed)

    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
    elapsed = time.perf_counter() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    image = result.images[0]
    image.save(out)

    manifest_path = out.with_suffix(".json")
    write_manifest(
        manifest_path,
        {
            "phase": "phase1_generate",
            "candidate": key,
            "model": model_id,
            "source_repo": candidate["repo"],
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": args.guidance_scale,
            "device": device.type,
            "dtype": str(dtype),
            "elapsed_seconds": elapsed,
            "image": str(out),
        },
    )
    print(out)
    print(manifest_path)


if __name__ == "__main__":
    main()
