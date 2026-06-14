#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from diffusers import DPMSolverMultistepScheduler
from PIL import Image, ImageDraw

from phase4_export_watch_prompt_assets import load_presets
from research_common import (
    ROOT,
    pick_torch_device,
    require_diffusion_stack,
    resolve_model_path,
    select_candidate,
    torch_dtype_for_device,
    write_manifest,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_sheet(entries: list[dict], out_path: Path, cell: int, label_height: int = 34) -> None:
    if not entries:
        return
    columns = min(4, len(entries))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        image = Image.open(entry["image"]).convert("RGB").resize((cell, cell), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        label = f'{entry["key"]} s{entry["seed"]} g{entry["guidance_scale"]:g}'
        draw.text((x + 4, y + cell + 8), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Mac-side reference grid for the current small Diffusers model."
    )
    parser.add_argument("--candidate", default="segmind_tiny_sd")
    parser.add_argument("--model", default=None)
    parser.add_argument("--presets", default=str(ROOT / "configs" / "sd_watch_presets.json"))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--guidance-scales", default="4,8,10")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    presets = load_presets(args.presets)[: args.limit]
    seeds = parse_int_list(args.seeds)
    guidance_scales = parse_float_list(args.guidance_scales)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / "student_reference" / key

    torch, diffusers = require_diffusion_stack()
    device = pick_torch_device(torch)
    dtype = torch_dtype_for_device(torch, device)
    pipeline_cls = getattr(diffusers, candidate["pipeline"], None)
    if pipeline_cls is None:
        raise SystemExit(f"diffusers does not expose {candidate['pipeline']}")

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    generator_device = "cpu" if device.type == "mps" else device.type
    entries: list[dict] = []
    started_at = time.perf_counter()
    for preset in presets:
        for seed in seeds:
            for guidance_scale in guidance_scales:
                generator = torch.Generator(device=generator_device).manual_seed(seed)
                image_path = out_dir / "images" / f'{preset["key"]}_seed{seed}_g{guidance_scale:g}.png'
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with torch.inference_mode():
                    result = pipe(
                        prompt=preset["prompt"],
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                result.images[0].save(image_path)
                entries.append(
                    {
                        "key": preset["key"],
                        "title": preset["title"],
                        "prompt": preset["prompt"],
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "steps": args.steps,
                        "image": str(image_path),
                    }
                )
                print(image_path)

    sheet_path = out_dir / "contact_sheet.png"
    make_sheet(entries, sheet_path, cell=args.width)
    manifest_path = out_dir / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "phase": "student_reference_grid",
            "candidate": key,
            "model": model_id,
            "device": device.type,
            "dtype": str(dtype),
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seeds": seeds,
            "guidance_scales": guidance_scales,
            "elapsed_seconds": time.perf_counter() - started_at,
            "contact_sheet": str(sheet_path),
            "entries": entries,
        },
    )
    print(sheet_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
