#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from make_watch_preview_comparison import upscale_2x_bicubic_like_watch, unsharp_like_watch


ROOT = Path(__file__).resolve().parents[1]


class PipelineSeededRandom:
    def __init__(self, seed: int):
        self.state = seed if seed != 0 else 0x9E3779B97F4A7C15

    def next_uint64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
        return (value ^ (value >> 31)) & 0xFFFF_FFFF_FFFF_FFFF

    def uniform(self) -> float:
        value = self.next_uint64() >> 40
        return float(value) / float(0x0100_0000)

    def normal(self) -> float:
        u1 = max(self.uniform(), 0.000001)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def fnv1a(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def load_presets(path: Path, keys: list[str] | None) -> list[dict[str, Any]]:
    presets = json.loads(path.read_text())
    if not keys:
        return presets
    by_key = {item["key"]: item for item in presets}
    missing = [key for key in keys if key not in by_key]
    if missing:
        known = ", ".join(by_key)
        raise SystemExit(f"unknown prompt keys {missing}; known keys: {known}")
    return [by_key[key] for key in keys]


def make_guidance_embedding(torch, guidance_scale: float, embedding_dim: int, device, dtype):
    w = torch.tensor([guidance_scale - 1.0], device=device, dtype=torch.float32) * 1000.0
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=device, dtype=torch.float32)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
    emb = w[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb.to(dtype=dtype)


def make_watch_latents(torch, shape: tuple[int, ...], seed: int, prompt_key: str, device, dtype):
    rng = PipelineSeededRandom(seed ^ fnv1a(prompt_key))
    values = np.array([rng.normal() for _ in range(int(np.prod(shape)))], dtype=np.float32)
    values = np.clip(values, -6.0, 6.0).reshape(shape)
    return torch.from_numpy(values).to(device=device, dtype=dtype), rng


def watch_step(
    torch,
    scheduler,
    latents,
    noise_pred,
    timestep: int,
    prev_timestep: int,
    is_final_step: bool,
    rng: PipelineSeededRandom,
):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=latents.device, dtype=torch.float32)
    beta_prod_t = 1.0 - alpha_prod_t
    alpha_prod_prev = scheduler.alphas_cumprod[prev_timestep].to(device=latents.device, dtype=torch.float32)
    beta_prod_prev = 1.0 - alpha_prod_prev
    c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep)
    c_skip = torch.as_tensor(c_skip, device=latents.device, dtype=torch.float32)
    c_out = torch.as_tensor(c_out, device=latents.device, dtype=torch.float32)

    latents_f32 = latents.to(torch.float32)
    noise_f32 = noise_pred.to(torch.float32)
    predicted_original = (latents_f32 - beta_prod_t.sqrt() * noise_f32) / torch.clamp(alpha_prod_t.sqrt(), min=0.000001)
    if scheduler.config.thresholding:
        predicted_original = scheduler._threshold_sample(predicted_original)
    elif scheduler.config.clip_sample:
        predicted_original = predicted_original.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    denoised = c_out * predicted_original + c_skip * latents_f32

    if is_final_step:
        return denoised.to(dtype=latents.dtype)

    noise_values = np.array([rng.normal() for _ in range(int(np.prod(tuple(latents.shape))))], dtype=np.float32)
    noise_values = np.clip(noise_values, -6.0, 6.0).reshape(tuple(latents.shape))
    noise = torch.from_numpy(noise_values).to(device=latents.device, dtype=torch.float32)
    updated = alpha_prod_prev.sqrt() * denoised + beta_prod_prev.sqrt() * noise
    return updated.to(dtype=latents.dtype)


def decoded_to_image(decoded):
    from PIL import Image

    values = decoded.detach().cpu().float().numpy()[0, :3]
    rgb = np.clip((values / 2.0 + 0.5) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")


def make_watch_preview(image, mode: str):
    if mode == "none":
        return None
    if mode == "sharp2x":
        return unsharp_like_watch(upscale_2x_bicubic_like_watch(image), amount=0.45)
    raise ValueError(f"unsupported preview mode: {mode}")


def tensor_stats(values) -> dict[str, float]:
    flat = values.detach().cpu().float().numpy().reshape(-1)
    finite = flat[np.isfinite(flat)]
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
    }


def make_contact_sheet(results: list[dict[str, Any]], columns: int, scale: int, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile = max(int(result.get("display_size", result["size"])) for result in results) * scale
    label_h = 36
    pad = 8
    columns = max(1, min(columns, len(results)))
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile + (columns + 1) * pad, rows * (tile + label_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    for index, result in enumerate(results):
        col = index % columns
        row = index // columns
        x = pad + col * (tile + pad)
        y = pad + row * (tile + label_h + pad)
        image = Image.open(result.get("preview_path") or result["path"]).convert("RGB")
        image = image.resize((image.width * scale, image.height * scale), nearest)
        draw.text((x, y), f"{result['key']} s{result['seed']} g{result['guidance_scale']:g}", fill=(0, 0, 0), font=font)
        draw.text((x, y + 13), f"rms {result['decoded_stats']['rms']:.3f} {result['elapsed_seconds']:.1f}s", fill=(80, 80, 80), font=font)
        sheet.paste(image, (x + (tile - image.width) // 2, y + label_h + (tile - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Watch-style LCM loop on CUDA for seed/prompt sweeps.")
    parser.add_argument("--model-id", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--presets", type=Path, default=ROOT / "configs" / "lcm64_watch_presets.json")
    parser.add_argument("--prompt-keys", nargs="*", default=["cat", "cat_face", "cat_logo", "orange_cat", "sitting_cat", "white_cat"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--guidance-scales", nargs="*", type=float, default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--preview-mode", choices=["none", "sharp2x"], default="none")
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import torch
    from diffusers import LatentConsistencyModelPipeline

    if args.size % 8 != 0:
        raise SystemExit("--size must be divisible by 8")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device("cuda")
    presets = load_presets(args.presets, args.prompt_keys)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "previews"
    if args.preview_mode != "none":
        preview_dir.mkdir(parents=True, exist_ok=True)

    pipe = LatentConsistencyModelPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.steps, device=device)
    time_cond_dim = int(getattr(pipe.unet.config, "time_cond_proj_dim", 256))

    prompts = [preset["prompt"] for preset in presets]
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    latent_shape = (1, 4, args.size // 8, args.size // 8)
    results: list[dict[str, Any]] = []
    guidance_scales = args.guidance_scales if args.guidance_scales else [args.guidance_scale]
    for guidance_scale in guidance_scales:
        timestep_cond = make_guidance_embedding(torch, guidance_scale, time_cond_dim, device, dtype)
        guidance_text = f"{guidance_scale:g}".replace(".", "p")
        for preset_index, preset in enumerate(presets):
            embedding = prompt_embeds[preset_index : preset_index + 1]
            for seed in args.seeds:
                start = time.perf_counter()
                latents, rng = make_watch_latents(torch, latent_shape, seed, preset["key"], device, dtype)
                with torch.inference_mode():
                    for step_index, timestep_tensor in enumerate(pipe.scheduler.timesteps):
                        timestep = int(timestep_tensor.item())
                        if step_index + 1 < len(pipe.scheduler.timesteps):
                            prev_timestep = int(pipe.scheduler.timesteps[step_index + 1].item())
                        else:
                            prev_timestep = timestep
                        noise_pred = pipe.unet(
                            latents,
                            timestep_tensor[None].to(device=device),
                            encoder_hidden_states=embedding,
                            timestep_cond=timestep_cond,
                            return_dict=False,
                        )[0]
                        latents = watch_step(
                            torch,
                            pipe.scheduler,
                            latents,
                            noise_pred,
                            timestep,
                            prev_timestep,
                            is_final_step=step_index == len(pipe.scheduler.timesteps) - 1,
                            rng=rng,
                        )
                    decoded = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(dtype=dtype), return_dict=False)[0]
                elapsed = time.perf_counter() - start
                image = decoded_to_image(decoded)
                filename = f"{preset['key']}_seed{seed}_{args.size}px_{args.steps}st_g{guidance_text}_watch.png"
                path = image_dir / filename
                image.save(path)
                preview = make_watch_preview(image, args.preview_mode)
                preview_path = None
                display_size = args.size
                if preview is not None:
                    preview_path = preview_dir / filename
                    preview.save(preview_path)
                    display_size = preview.width
                decoded_stats = tensor_stats(decoded)
                result = {
                    "key": preset["key"],
                    "title": preset["title"],
                    "prompt": preset["prompt"],
                    "seed": seed,
                    "size": args.size,
                    "display_size": display_size,
                    "steps": args.steps,
                    "guidance_scale": guidance_scale,
                    "elapsed_seconds": elapsed,
                    "decoded_stats": decoded_stats,
                    "path": str(path),
                    "preview_mode": args.preview_mode,
                    "preview_path": str(preview_path) if preview_path else None,
                }
                results.append(result)
                print(f"{preset['key']} seed={seed} guidance={guidance_scale:g} decoded_rms={decoded_stats['rms']:.4f} elapsed={elapsed:.2f}s", flush=True)

    contact_sheet = args.out_dir / "contact_sheet.png"
    make_contact_sheet(results, args.columns, args.scale, contact_sheet)
    manifest = {
        "phase": "cloud_lcm_watch_style_probe",
        "model_id": args.model_id,
        "dtype": args.dtype,
        "presets": str(args.presets),
        "prompt_keys": [preset["key"] for preset in presets],
        "seeds": args.seeds,
        "size": args.size,
        "steps": args.steps,
        "guidance_scales": guidance_scales,
        "preview_mode": args.preview_mode,
        "latent_shape": list(latent_shape),
        "contact_sheet": str(contact_sheet),
        "results": results,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(contact_sheet)


if __name__ == "__main__":
    main()
