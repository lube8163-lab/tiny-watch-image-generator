#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_export_unet import make_guidance_embedding
from research_common import ROOT, require_diffusion_stack, resolve_model_path, select_candidate, write_manifest


DEFAULT_PRESETS = [
    {
        "key": "cat",
        "title": "Cat",
        "prompt": "cute anime cat, clean illustration",
        "aliases": ["cat", "kitty", "neko", "ねこ", "猫"],
    },
    {
        "key": "dog",
        "title": "Dog",
        "prompt": "cute anime dog, clean illustration",
        "aliases": ["dog", "puppy", "inu", "いぬ", "犬"],
    },
    {
        "key": "girl",
        "title": "Girl",
        "prompt": "anime girl portrait, clean illustration",
        "aliases": ["girl", "person", "portrait", "anime", "女の子"],
    },
    {
        "key": "castle",
        "title": "Castle",
        "prompt": "fantasy castle, anime background art",
        "aliases": ["castle", "fantasy", "shiro", "城"],
    },
    {
        "key": "flower",
        "title": "Flower",
        "prompt": "beautiful flower, clean anime illustration",
        "aliases": ["flower", "rose", "hana", "花"],
    },
    {
        "key": "mountain",
        "title": "Mountain",
        "prompt": "mountain landscape, anime background art",
        "aliases": ["mountain", "landscape", "yama", "山"],
    },
    {
        "key": "robot",
        "title": "Robot",
        "prompt": "small friendly robot, clean anime illustration",
        "aliases": ["robot", "mecha", "ロボット"],
    },
    {
        "key": "star",
        "title": "Star",
        "prompt": "starry night sky, anime background art",
        "aliases": ["star", "sky", "night", "星", "夜空"],
    },
]


def load_presets(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_PRESETS
    return json.loads(Path(path).read_text())


def validate_dimensions(latent_height: int, latent_width: int, output_height: int, output_width: int) -> None:
    dimensions = [latent_height, latent_width, output_height, output_width]
    if any(value <= 0 for value in dimensions):
        raise SystemExit("latent and output dimensions must be positive")
    if output_height != latent_height * 8 or output_width != latent_width * 8:
        raise SystemExit("output dimensions must be 8x the latent dimensions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tiny Watch prompt assets for preset-based LCM txt2img.")
    parser.add_argument("--candidate", default="lcm_dreamshaper_v7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--presets", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--output-height", type=int, default=64)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    validate_dimensions(args.latent_height, args.latent_width, args.output_height, args.output_width)

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "dist" / key / f"watch_txt2img_{args.output_width}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch, diffusers = require_diffusion_stack()
    pipeline_cls = getattr(diffusers, candidate["pipeline"], None)
    if pipeline_cls is None:
        raise SystemExit(f"diffusers does not expose {candidate['pipeline']}; update diffusers.")

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cpu")

    presets = load_presets(args.presets)
    prompts = [item["prompt"] for item in presets]
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to("cpu", dtype=torch.float16).contiguous()

        time_cond_dim = int(getattr(pipe.unet.config, "time_cond_proj_dim", 256))
        timestep_cond = make_guidance_embedding(
            torch,
            torch.tensor([args.guidance_scale - 1.0], dtype=torch.float32),
            time_cond_dim,
            torch.float32,
        ).to(torch.float16).contiguous()

    pipe.scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = [int(t) for t in pipe.scheduler.timesteps.tolist()]
    alphas = pipe.scheduler.alphas_cumprod

    steps = []
    for index, timestep in enumerate(timesteps):
        prev_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else timestep
        alpha_t = float(alphas[timestep])
        beta_t = 1.0 - alpha_t
        alpha_prev = float(alphas[prev_timestep])
        beta_prev = 1.0 - alpha_prev
        c_skip, c_out = pipe.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        steps.append(
            {
                "timestep": timestep,
                "sqrtAlpha": alpha_t**0.5,
                "sqrtBeta": beta_t**0.5,
                "sqrtAlphaPrev": alpha_prev**0.5,
                "sqrtBetaPrev": beta_prev**0.5,
                "cSkip": float(c_skip),
                "cOut": float(c_out),
            }
        )

    embeddings_path = out_dir / "prompt_embeddings_f16.bin"
    timestep_cond_path = out_dir / "timestep_cond_f16.bin"
    prompts_path = out_dir / "prompt_presets.json"
    scheduler_path = out_dir / "lcm_scheduler.json"
    manifest_path = out_dir / "manifest.json"

    embeddings_path.write_bytes(prompt_embeds.numpy().tobytes())
    timestep_cond_path.write_bytes(timestep_cond.numpy().tobytes())
    prompts_path.write_text(
        json.dumps(
            {
                "embeddingShape": list(prompt_embeds.shape),
                "embeddingDtype": "float16",
                "presets": presets,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    scheduler_path.write_text(
        json.dumps(
            {
                "steps": steps,
                "guidanceScale": args.guidance_scale,
                "timestepCondShape": list(timestep_cond.shape),
                "latentShape": [1, 4, args.latent_height, args.latent_width],
                "decodedShape": [1, 3, args.output_height, args.output_width],
                "predictionType": pipe.scheduler.config.prediction_type,
            },
            indent=2,
        )
        + "\n"
    )
    write_manifest(
        manifest_path,
        {
            "phase": "phase4_export_watch_prompt_assets",
            "candidate": key,
            "model": model_id,
            "prompt_count": len(presets),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "latent_shape": [1, 4, args.latent_height, args.latent_width],
            "decoded_shape": [1, 3, args.output_height, args.output_width],
            "prompt_embeddings": str(embeddings_path),
            "prompt_embeddings_bytes": embeddings_path.stat().st_size,
            "timestep_cond": str(timestep_cond_path),
            "timestep_cond_bytes": timestep_cond_path.stat().st_size,
            "prompts": str(prompts_path),
            "scheduler": str(scheduler_path),
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
