#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from make_watch_preview_comparison import upscale_2x_bicubic_like_watch, unsharp_like_watch
from research_common import ROOT, require_diffusion_stack, resolve_model_path, select_candidate, write_manifest


DEFAULT_ASSET_DIR = ROOT / "watchos_example" / "WatchPipelineSmokeApp" / "LCMAssets"
DEFAULT_MODEL_DIR = ROOT / "dist" / "lcm_dreamshaper_v7"


class PipelineSeededRandom:
    def __init__(self, seed: int):
        self.state = seed if seed != 0 else 0x9E3779B97F4A7C15

    def normal(self) -> float:
        u1 = max(self.uniform(), 0.000001)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def uniform(self) -> float:
        value = self.next_uint64() >> 40
        return float(value) / float(0x0100_0000)

    def next_uint64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
        return (value ^ (value >> 31)) & 0xFFFF_FFFF_FFFF_FFFF


def fnv1a(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def tensor_stats(values: np.ndarray) -> dict[str, float]:
    flat = values.astype(np.float32, copy=False).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "rms": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
    }


def clipped_count(decoded: np.ndarray) -> int:
    rgb = decoded.astype(np.float32, copy=False).reshape(-1)
    return int(np.count_nonzero((rgb < -1.0) | (rgb > 1.0)))


def format_stats(stats: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in stats.items())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_assets(asset_dir: Path, prompt_key: str) -> dict[str, Any]:
    preset_file = load_json(asset_dir / "prompt_presets.json")
    scheduler = load_json(asset_dir / "lcm_scheduler.json")
    presets = preset_file["presets"]
    try:
        preset_index, preset = next((index, item) for index, item in enumerate(presets) if item["key"] == prompt_key)
    except StopIteration as exc:
        known = ", ".join(item["key"] for item in presets)
        raise SystemExit(f"unknown prompt key '{prompt_key}'. Known keys: {known}") from exc

    prompt_embeddings = np.fromfile(asset_dir / "prompt_embeddings_f16.bin", dtype=np.float16)
    prompt_embeddings = prompt_embeddings.reshape(preset_file["embeddingShape"])
    timestep_cond = np.fromfile(asset_dir / "timestep_cond_f16.bin", dtype=np.float16)
    timestep_cond = timestep_cond.reshape(scheduler["timestepCondShape"])

    return {
        "preset_file": preset_file,
        "scheduler": scheduler,
        "preset_index": preset_index,
        "preset": preset,
        "prompt_embedding": prompt_embeddings[preset_index : preset_index + 1],
        "timestep_cond": timestep_cond,
    }


def make_timestep_cond_for_guidance(guidance_scale: float, shape: list[int]) -> np.ndarray:
    from phase3_export_unet import make_guidance_embedding

    torch, _ = require_diffusion_stack()
    if len(shape) != 2 or shape[0] != 1:
        raise SystemExit(f"unexpected timestep_cond shape: {shape}")
    embedding = make_guidance_embedding(
        torch,
        torch.tensor([guidance_scale - 1.0], dtype=torch.float32),
        int(shape[1]),
        torch.float32,
    )
    return embedding.to(torch.float16).cpu().numpy()


def parse_shape(text: str) -> list[int]:
    try:
        shape = [int(part) for part in text.lower().replace("x", ",").split(",") if part]
    except ValueError as exc:
        raise SystemExit(f"invalid shape {text!r}; expected e.g. 1,4,16,16") from exc
    if not shape or any(value <= 0 for value in shape):
        raise SystemExit(f"invalid shape {text!r}; expected positive dimensions")
    return shape


def make_watch_latents(shape: list[int], seed: int, prompt_key: str) -> tuple[np.ndarray, PipelineSeededRandom]:
    rng = PipelineSeededRandom(seed ^ fnv1a(prompt_key))
    values = np.array([rng.normal() for _ in range(int(np.prod(shape)))], dtype=np.float32)
    values = np.clip(values, -6.0, 6.0)
    return values.reshape(shape), rng


def update_lcm_latents(
    latents: np.ndarray,
    noise: np.ndarray,
    step: dict[str, float],
    is_final_step: bool,
    rng: PipelineSeededRandom,
) -> np.ndarray:
    latents_flat = latents.astype(np.float32, copy=False).reshape(-1)
    noise_flat = noise.astype(np.float32, copy=False).reshape(-1)
    output = np.empty_like(latents_flat)
    for index, latent in enumerate(latents_flat):
        predicted_original = (float(latent) - float(step["sqrtBeta"]) * float(noise_flat[index])) / max(
            float(step["sqrtAlpha"]),
            0.000001,
        )
        denoised = float(step["cOut"]) * predicted_original + float(step["cSkip"]) * float(latent)
        if is_final_step:
            output[index] = denoised
        else:
            output[index] = float(step["sqrtAlphaPrev"]) * denoised + float(step["sqrtBetaPrev"]) * max(
                -6.0,
                min(6.0, rng.normal()),
            )
    return output.reshape(latents.shape)


def decoded_to_image(decoded: np.ndarray) -> Image.Image:
    values = decoded.astype(np.float32, copy=False)
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] < 3:
        raise SystemExit(f"unexpected decoded shape: {values.shape}")
    rgb = values[0, :3]
    rgb = np.clip((rgb / 2.0 + 0.5) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")


def make_watch_preview(image: Image.Image, mode: str) -> Image.Image | None:
    if mode == "none":
        return None
    if mode == "sharp2x":
        return unsharp_like_watch(upscale_2x_bicubic_like_watch(image), amount=0.45)
    raise ValueError(f"unsupported preview mode: {mode}")


def run_watch_loop(
    *,
    scheduler: dict[str, Any],
    prompt_key: str,
    seed: int,
    predict_noise,
    decode_latents,
) -> dict[str, Any]:
    latents, rng = make_watch_latents(scheduler["latentShape"], seed, prompt_key)
    step_reports = []
    started = time.perf_counter()

    for step_index, step in enumerate(scheduler["steps"]):
        step_started = time.perf_counter()
        noise = predict_noise(latents, step)
        noise_stats = tensor_stats(noise)
        latents = update_lcm_latents(
            latents,
            noise,
            step,
            is_final_step=step_index == len(scheduler["steps"]) - 1,
            rng=rng,
        )
        step_reports.append(
            {
                "index": step_index + 1,
                "timestep": int(step["timestep"]),
                "elapsed_seconds": time.perf_counter() - step_started,
                "noise_stats": noise_stats,
                "latents_stats": tensor_stats(latents),
            }
        )

    decoded_started = time.perf_counter()
    decoded = decode_latents(latents)
    decoder_elapsed = time.perf_counter() - decoded_started
    decoded_rgb = decoded[:, :3]
    total_channels = int(np.prod(decoded_rgb.shape))
    return {
        "latents": latents,
        "decoded": decoded_rgb,
        "steps": step_reports,
        "decoder_elapsed_seconds": decoder_elapsed,
        "total_elapsed_seconds": time.perf_counter() - started,
        "final_stats": tensor_stats(latents),
        "decoded_stats": tensor_stats(decoded_rgb),
        "clipped_channels": clipped_count(decoded_rgb),
        "total_channels": total_channels,
    }


def run_coreml_reference(
    *,
    unet_packages: list[Path],
    decoder_package: Path,
    assets: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    import coremltools as ct

    unets = [
        ct.models.MLModel(
            str(unet_package),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        for unet_package in unet_packages
    ]
    unet_input_names = [
        {item.name for item in unet.get_spec().description.input}
        for unet in unets
    ]
    decoder = ct.models.MLModel(
        str(decoder_package),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    prompt_embedding = assets["prompt_embedding"].astype(np.float16, copy=False)
    timestep_cond = assets["timestep_cond"].astype(np.float16, copy=False)
    scheduler = assets["scheduler"]

    def predict_noise(latents: np.ndarray, step: dict[str, float]) -> np.ndarray:
        values = {
            "sample": latents.astype(np.float16),
            "timestep": np.array([float(step["timestep"])], dtype=np.float16),
            "encoder_hidden_states": prompt_embedding,
            "timestep_cond": timestep_cond,
        }
        output = {}
        for unet, input_names in zip(unets, unet_input_names):
            filtered = {name: value for name, value in values.items() if name in input_names}
            output = unet.predict(filtered)
            values.update(output)
        return np.asarray(output["noise_pred"], dtype=np.float32)

    def decode_latents(latents: np.ndarray) -> np.ndarray:
        output = decoder.predict({"latents": latents.astype(np.float16)})
        return np.asarray(output["decoded"], dtype=np.float32)

    return run_watch_loop(
        scheduler=scheduler,
        prompt_key=assets["preset"]["key"],
        seed=seed,
        predict_noise=predict_noise,
        decode_latents=decode_latents,
    )


def run_torch_watch_reference(
    *,
    candidate_name: str,
    model_override: str | None,
    assets: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    torch, diffusers = require_diffusion_stack()
    key, candidate = select_candidate(candidate_name)
    model_id = resolve_model_path(key, candidate, model_override, local_files_only=True)
    pipeline_cls = getattr(diffusers, candidate["pipeline"])
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cpu")
    pipe.unet.eval()
    pipe.vae.eval()

    prompt_embedding_np = assets["prompt_embedding"].astype(np.float16, copy=False)
    timestep_cond_np = assets["timestep_cond"].astype(np.float16, copy=False)
    scheduler = assets["scheduler"]

    def predict_noise(latents: np.ndarray, step: dict[str, float]) -> np.ndarray:
        with torch.inference_mode():
            sample = torch.from_numpy(latents.astype(np.float16)).to(dtype=torch.float32)
            timestep = torch.tensor([float(step["timestep"])], dtype=torch.float32)
            prompt_embedding = torch.from_numpy(prompt_embedding_np).to(dtype=torch.float32)
            timestep_cond = torch.from_numpy(timestep_cond_np).to(dtype=torch.float32)
            output = pipe.unet(
                sample,
                timestep,
                encoder_hidden_states=prompt_embedding,
                timestep_cond=timestep_cond,
                return_dict=False,
            )[0]
        return output.detach().cpu().numpy().astype(np.float32)

    def decode_latents(latents: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            sample = torch.from_numpy(latents.astype(np.float16)).to(dtype=torch.float32)
            decoded = pipe.vae.decode(sample / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return decoded.detach().cpu().numpy().astype(np.float32)

    return run_watch_loop(
        scheduler=scheduler,
        prompt_key=assets["preset"]["key"],
        seed=seed,
        predict_noise=predict_noise,
        decode_latents=decode_latents,
    )


def run_torch_native_reference(
    *,
    candidate_name: str,
    model_override: str | None,
    assets: dict[str, Any],
    seed: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    torch, diffusers = require_diffusion_stack()
    key, candidate = select_candidate(candidate_name)
    model_id = resolve_model_path(key, candidate, model_override, local_files_only=True)
    pipeline_cls = getattr(diffusers, candidate["pipeline"])
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=assets["preset"]["prompt"],
            width=width,
            height=height,
            num_inference_steps=len(assets["scheduler"]["steps"]),
            guidance_scale=float(assets["scheduler"]["guidanceScale"]),
            generator=generator,
            output_type="latent",
        )
        latents = result.images.detach().cpu().numpy().astype(np.float32)
        decoded = pipe.vae.decode(result.images / pipe.vae.config.scaling_factor, return_dict=False)[0]
        decoded = decoded.detach().cpu().numpy().astype(np.float32)
    decoded_rgb = decoded[:, :3]
    return {
        "latents": latents,
        "decoded": decoded_rgb,
        "steps": [],
        "decoder_elapsed_seconds": None,
        "total_elapsed_seconds": time.perf_counter() - started,
        "final_stats": tensor_stats(latents),
        "decoded_stats": tensor_stats(decoded_rgb),
        "clipped_channels": clipped_count(decoded_rgb),
        "total_channels": int(np.prod(decoded_rgb.shape)),
    }


def write_outputs(
    *,
    out_dir: Path,
    mode: str,
    args: argparse.Namespace,
    assets: dict[str, Any],
    result: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{mode}.png"
    decoded_path = out_dir / f"{mode}_decoded.npy"
    latents_path = out_dir / f"{mode}_latents.npy"
    manifest_path = out_dir / f"{mode}.json"
    preview_path = out_dir / f"{mode}_preview_{args.preview_mode}.png"

    image = decoded_to_image(result["decoded"])
    image.save(image_path)
    preview = make_watch_preview(image, args.preview_mode)
    if preview is not None:
        preview.save(preview_path)
    np.save(decoded_path, result["decoded"])
    np.save(latents_path, result["latents"])

    payload = {
        "phase": "generate_lcm64_reference",
        "mode": mode,
        "prompt_key": assets["preset"]["key"],
        "prompt": assets["preset"]["prompt"],
        "seed": args.seed,
        "guidance_scale": float(assets["scheduler"]["guidanceScale"]),
        "steps": len(assets["scheduler"]["steps"]),
        "unet_package": args.unet_package,
        "decoder_package": args.decoder_package,
        "latent_shape": list(result["latents"].shape),
        "decoded_shape": list(result["decoded"].shape),
        "final_stats": result["final_stats"],
        "decoded_stats": result["decoded_stats"],
        "clipped_channels": result["clipped_channels"],
        "total_channels": result["total_channels"],
        "decoder_elapsed_seconds": result["decoder_elapsed_seconds"],
        "total_elapsed_seconds": result["total_elapsed_seconds"],
        "step_reports": result["steps"],
        "image": str(image_path),
        "preview_mode": args.preview_mode,
        "preview_image": str(preview_path) if preview is not None else None,
        "decoded_npy": str(decoded_path),
        "latents_npy": str(latents_path),
    }
    write_manifest(manifest_path, payload)

    print(f"{mode}: {image_path}")
    print(f"  final {format_stats(result['final_stats'])}")
    print(f"  decoded {format_stats(result['decoded_stats'])}")
    print(f"  clipped {result['clipped_channels']}/{result['total_channels']}")
    if preview is not None:
        print(f"  preview {preview_path}")
    print(f"  total {result['total_elapsed_seconds']:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LCM64 local references matching WatchPipelineSmokeApp settings.")
    parser.add_argument("--mode", choices=["coreml", "torch-watch", "torch-native", "all"], default="coreml")
    parser.add_argument("--candidate", default="lcm_dreamshaper_v7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--unet-package", nargs="+", default=None)
    parser.add_argument("--decoder-package", default=None)
    parser.add_argument("--prompt-key", default="cat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--latent-shape", default=None, help="Override scheduler latent shape, e.g. 1,4,16,16.")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--preview-mode", choices=["none", "sharp2x"], default="none")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    model_dir = Path(args.model_dir)
    unet_packages = (
        [Path(item) for item in args.unet_package]
        if args.unet_package
        else [model_dir / "unet_8x8_4bit.mlpackage"]
    )
    decoder_package = (
        Path(args.decoder_package)
        if args.decoder_package
        else model_dir / "vae_decoder_64x64_noattn_4bit.mlpackage"
    )
    args.unet_package = [str(item) for item in unet_packages]
    args.decoder_package = str(decoder_package)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / "watch_pipeline_reference" / f"lcm64_{args.prompt_key}_seed{args.seed}"
    assets = load_assets(asset_dir, args.prompt_key)
    if args.latent_shape:
        assets["scheduler"] = dict(assets["scheduler"])
        assets["scheduler"]["latentShape"] = parse_shape(args.latent_shape)
    if args.guidance_scale is not None:
        assets["scheduler"] = dict(assets["scheduler"])
        assets["scheduler"]["guidanceScale"] = args.guidance_scale
        assets["timestep_cond"] = make_timestep_cond_for_guidance(
            args.guidance_scale,
            assets["scheduler"]["timestepCondShape"],
        )

    modes = ["coreml", "torch-watch", "torch-native"] if args.mode == "all" else [args.mode]
    for mode in modes:
        if mode == "coreml":
            result = run_coreml_reference(
                unet_packages=unet_packages,
                decoder_package=decoder_package,
                assets=assets,
                seed=args.seed,
            )
        elif mode == "torch-watch":
            result = run_torch_watch_reference(
                candidate_name=args.candidate,
                model_override=args.model,
                assets=assets,
                seed=args.seed,
            )
        elif mode == "torch-native":
            result = run_torch_native_reference(
                candidate_name=args.candidate,
                model_override=args.model,
                assets=assets,
                seed=args.seed,
                width=args.width,
                height=args.height,
            )
        else:
            raise AssertionError(mode)
        write_outputs(out_dir=out_dir, mode=mode, args=args, assets=assets, result=result)


if __name__ == "__main__":
    main()
