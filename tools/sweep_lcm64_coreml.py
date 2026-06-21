#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from generate_lcm64_reference import (
    DEFAULT_ASSET_DIR,
    DEFAULT_MODEL_DIR,
    clipped_count,
    decoded_to_image,
    format_stats,
    load_json,
    make_watch_preview,
    parse_shape,
    run_watch_loop,
    tensor_stats,
    write_manifest,
)


def load_asset_bundle(asset_dir: Path) -> dict[str, Any]:
    preset_file = load_json(asset_dir / "prompt_presets.json")
    scheduler = load_json(asset_dir / "lcm_scheduler.json")
    prompt_embeddings = np.fromfile(asset_dir / "prompt_embeddings_f16.bin", dtype=np.float16)
    prompt_embeddings = prompt_embeddings.reshape(preset_file["embeddingShape"])
    timestep_cond = np.fromfile(asset_dir / "timestep_cond_f16.bin", dtype=np.float16)
    timestep_cond = timestep_cond.reshape(scheduler["timestepCondShape"])
    return {
        "preset_file": preset_file,
        "scheduler": scheduler,
        "prompt_embeddings": prompt_embeddings,
        "timestep_cond": timestep_cond,
    }


def make_timestep_cond_for_guidance(guidance_scale: float, shape: list[int]) -> np.ndarray:
    from phase3_export_unet import make_guidance_embedding
    from research_common import require_diffusion_stack

    torch, _ = require_diffusion_stack()
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError(f"unexpected timestep_cond shape: {shape}")
    embedding = make_guidance_embedding(
        torch,
        torch.tensor([guidance_scale - 1.0], dtype=torch.float32),
        int(shape[1]),
        torch.float32,
    )
    return embedding.to(torch.float16).cpu().numpy()


def asset_bundle_with_guidance(asset_bundle: dict[str, Any], guidance_scale: float) -> dict[str, Any]:
    adjusted = dict(asset_bundle)
    scheduler = copy.deepcopy(asset_bundle["scheduler"])
    scheduler["guidanceScale"] = guidance_scale
    adjusted["scheduler"] = scheduler
    adjusted["timestep_cond"] = make_timestep_cond_for_guidance(
        guidance_scale,
        scheduler["timestepCondShape"],
    )
    return adjusted


def resolve_presets(asset_bundle: dict[str, Any], keys: list[str] | None) -> list[tuple[int, dict[str, Any]]]:
    presets = asset_bundle["preset_file"]["presets"]
    if not keys:
        return list(enumerate(presets))
    by_key = {item["key"]: (index, item) for index, item in enumerate(presets)}
    missing = [key for key in keys if key not in by_key]
    if missing:
        known = ", ".join(item["key"] for item in presets)
        raise SystemExit(f"unknown prompt keys {missing}; known keys: {known}")
    return [by_key[key] for key in keys]


def resolve_pairs(asset_bundle: dict[str, Any], pairs: list[str]) -> list[tuple[int, dict[str, Any], int]]:
    presets = asset_bundle["preset_file"]["presets"]
    by_key = {item["key"]: (index, item) for index, item in enumerate(presets)}
    resolved: list[tuple[int, dict[str, Any], int]] = []
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"pair must be key:seed, got {pair!r}")
        key, seed_text = pair.rsplit(":", 1)
        if key not in by_key:
            known = ", ".join(item["key"] for item in presets)
            raise SystemExit(f"unknown prompt key {key!r}; known keys: {known}")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise SystemExit(f"pair seed must be an integer, got {pair!r}") from exc
        preset_index, preset = by_key[key]
        resolved.append((preset_index, preset, seed))
    return resolved


def load_coreml_models(unet_packages: list[Path], decoder_package: Path):
    import coremltools as ct

    unets = [
        ct.models.MLModel(
            str(path),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        for path in unet_packages
    ]
    input_names = [
        {item.name for item in unet.get_spec().description.input}
        for unet in unets
    ]
    decoder = ct.models.MLModel(
        str(decoder_package),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    return unets, input_names, decoder


def generate_one(
    *,
    preset_index: int,
    preset: dict[str, Any],
    seed: int,
    asset_bundle: dict[str, Any],
    unets,
    input_names,
    decoder,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    prompt_embedding = asset_bundle["prompt_embeddings"][preset_index : preset_index + 1].astype(np.float16)
    timestep_cond = asset_bundle["timestep_cond"].astype(np.float16)

    def predict_noise(latents: np.ndarray, step: dict[str, float]) -> np.ndarray:
        values = {
            "sample": latents.astype(np.float16),
            "timestep": np.array([float(step["timestep"])], dtype=np.float16),
            "encoder_hidden_states": prompt_embedding,
            "timestep_cond": timestep_cond,
        }
        output = {}
        for unet, names in zip(unets, input_names):
            output = unet.predict({name: value for name, value in values.items() if name in names})
            values.update(output)
        return np.asarray(output["noise_pred"], dtype=np.float32)

    def decode_latents(latents: np.ndarray) -> np.ndarray:
        return np.asarray(decoder.predict({"latents": latents.astype(np.float16)})["decoded"], dtype=np.float32)

    result = run_watch_loop(
        scheduler=asset_bundle["scheduler"],
        prompt_key=preset["key"],
        seed=seed,
        predict_noise=predict_noise,
        decode_latents=decode_latents,
    )
    return {
        "preset_index": preset_index,
        "preset_key": preset["key"],
        "title": preset["title"],
        "prompt": preset["prompt"],
        "seed": seed,
        "guidance_scale": (
            float(guidance_scale)
            if guidance_scale is not None
            else float(asset_bundle["scheduler"]["guidanceScale"])
        ),
        "final_stats": result["final_stats"],
        "decoded_stats": result["decoded_stats"],
        "clipped_channels": result["clipped_channels"],
        "total_channels": result["total_channels"],
        "image": decoded_to_image(result["decoded"]),
    }


def make_contact_sheet(results: list[dict[str, Any]], columns: int, scale: int, path: Path) -> None:
    tile = max(result["display_image"].width for result in results) * scale
    label_h = 34
    pad = 8
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * pad, rows * (tile + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        row = index // columns
        col = index % columns
        x = pad + col * (tile + pad)
        y = pad + row * (tile + label_h + pad)
        label = f"{result['preset_key']} s{result['seed']} g{result['guidance_scale']:g}"
        rms = result["decoded_stats"]["rms"]
        clipped = result["clipped_channels"]
        draw.text((x, y), label[:30], fill=(0, 0, 0))
        draw.text((x, y + 12), f"rms {rms:.3f} clip {clipped}", fill=(80, 80, 80))
        image = result["display_image"].resize(
            (result["display_image"].width * scale, result["display_image"].height * scale),
            Image.Resampling.NEAREST,
        )
        sheet.paste(image, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_individual_images(results: list[dict[str, Any]], out_dir: Path, preview_mode: str) -> None:
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    if preview_mode != "none":
        preview_dir.mkdir(parents=True, exist_ok=True)
    include_guidance = len({result["guidance_scale"] for result in results}) > 1
    for result in results:
        guidance_suffix = ""
        if include_guidance:
            guidance = f"{float(result['guidance_scale']):g}".replace(".", "p")
            guidance_suffix = f"_g{guidance}"
        path = image_dir / f"{result['preset_key']}_seed{result['seed']}{guidance_suffix}.png"
        result["image"].save(path)
        result["image_path"] = str(path)
        result["preview_mode"] = preview_mode
        result["preview_path"] = None
        result["display_image"] = result["image"]
        preview = make_watch_preview(result["image"], preview_mode)
        if preview is not None:
            preview_path = preview_dir / path.name
            preview.save(preview_path)
            result["preview_path"] = str(preview_path)
            result["display_image"] = preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LCM64 Core ML prompt presets and seeds.")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--unet-package", nargs="+", default=None)
    parser.add_argument("--decoder-package", default=None)
    parser.add_argument("--prompt-keys", nargs="*", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 7, 11, 42, 1234])
    parser.add_argument("--latent-shape", default=None, help="Override scheduler latent shape, e.g. 1,4,16,16.")
    parser.add_argument("--guidance-scales", nargs="+", type=float, default=None)
    parser.add_argument("--preview-mode", choices=["none", "sharp2x"], default="none")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    model_dir = Path(args.model_dir)
    unet_packages = (
        [Path(item) for item in args.unet_package]
        if args.unet_package
        else [model_dir / "unet_8x8_6bit.mlpackage"]
    )
    decoder_package = (
        Path(args.decoder_package)
        if args.decoder_package
        else model_dir / "vae_decoder_64x64_noattn_4bit.mlpackage"
    )
    out_dir = Path(args.out_dir)
    asset_bundle = load_asset_bundle(asset_dir)
    if args.latent_shape:
        asset_bundle["scheduler"] = dict(asset_bundle["scheduler"])
        asset_bundle["scheduler"]["latentShape"] = parse_shape(args.latent_shape)
    presets = resolve_presets(asset_bundle, args.prompt_keys)
    pair_presets = resolve_pairs(asset_bundle, args.pairs) if args.pairs else None
    unets, input_names, decoder = load_coreml_models(unet_packages, decoder_package)

    results: list[dict[str, Any]] = []
    guidance_scales = args.guidance_scales or [float(asset_bundle["scheduler"]["guidanceScale"])]
    for guidance_scale in guidance_scales:
        guidance_bundle = (
            asset_bundle
            if guidance_scale == float(asset_bundle["scheduler"]["guidanceScale"])
            else asset_bundle_with_guidance(asset_bundle, guidance_scale)
        )
        jobs = pair_presets or [
            (preset_index, preset, seed)
            for preset_index, preset in presets
            for seed in args.seeds
        ]
        for preset_index, preset, seed in jobs:
            result = generate_one(
                preset_index=preset_index,
                preset=preset,
                seed=seed,
                asset_bundle=guidance_bundle,
                unets=unets,
                input_names=input_names,
                decoder=decoder,
                guidance_scale=guidance_scale,
            )
            results.append(result)
            decoded_stats = format_stats(result["decoded_stats"])
            print(
                f"{result['preset_key']} seed={seed} guidance={guidance_scale:g} "
                f"decoded {decoded_stats} clipped={result['clipped_channels']}"
            )

    write_individual_images(results, out_dir, args.preview_mode)
    contact_sheet = out_dir / "contact_sheet.png"
    make_contact_sheet(results, args.columns, args.scale, contact_sheet)
    manifest_results = []
    for result in results:
        manifest_result = dict(result)
        manifest_result.pop("image")
        manifest_result.pop("display_image")
        manifest_results.append(manifest_result)

    write_manifest(
        out_dir / "manifest.json",
        {
            "phase": "sweep_lcm64_coreml",
            "asset_dir": str(asset_dir),
            "unet_packages": [str(path) for path in unet_packages],
            "decoder_package": str(decoder_package),
            "seeds": args.seeds,
            "latent_shape": asset_bundle["scheduler"]["latentShape"],
            "guidance_scales": guidance_scales,
            "prompt_keys": [preset["key"] for _, preset in presets],
            "pairs": args.pairs,
            "contact_sheet": str(contact_sheet),
            "results": manifest_results,
        },
    )
    print(contact_sheet)


if __name__ == "__main__":
    main()
