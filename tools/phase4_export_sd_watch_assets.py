#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusers import DPMSolverMultistepScheduler
from phase4_export_watch_prompt_assets import DEFAULT_PRESETS, load_presets
from research_common import ROOT, require_diffusion_stack, resolve_model_path, select_candidate, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Watch assets for a small SD/DDIM preset txt2img path.")
    parser.add_argument("--candidate", default="segmind_tiny_sd")
    parser.add_argument("--model", default=None)
    parser.add_argument("--presets", default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "dist" / key / "watch_sd_txt2img_128"
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
        uncond_embeds, _ = pipe.encode_prompt(
            prompt=[""],
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        uncond_embeds = uncond_embeds.to("cpu", dtype=torch.float16).contiguous()

    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = [int(t) for t in scheduler.timesteps.tolist()]
    sigmas = [float(s) for s in scheduler.sigmas.tolist()]

    embeddings_path = out_dir / "prompt_embeddings_f16.bin"
    uncond_embeddings_path = out_dir / "uncond_embedding_f16.bin"
    prompts_path = out_dir / "prompt_presets.json"
    scheduler_path = out_dir / "sd_ddim_scheduler.json"
    manifest_path = out_dir / "manifest.json"

    embeddings_path.write_bytes(prompt_embeds.numpy().tobytes())
    uncond_embeddings_path.write_bytes(uncond_embeds.numpy().tobytes())
    prompts_path.write_text(
        json.dumps(
            {
                "embeddingShape": list(prompt_embeds.shape),
                "uncondEmbeddingShape": list(uncond_embeds.shape),
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
                "timesteps": timesteps,
                "sigmas": sigmas,
                "latentShape": [1, 4, 16, 16],
                "decodedShape": [1, 3, 128, 128],
                "predictionType": "epsilon",
                "scheduler": "DPMSolverMultistepScheduler",
                "algorithmType": scheduler.config.algorithm_type,
                "solverOrder": scheduler.config.solver_order,
                "solverType": scheduler.config.solver_type,
                "lowerOrderFinal": scheduler.config.lower_order_final,
                "finalSigmasType": scheduler.config.final_sigmas_type,
                "guidanceScale": args.guidance_scale,
            },
            indent=2,
        )
        + "\n"
    )
    write_manifest(
        manifest_path,
        {
            "phase": "phase4_export_sd_watch_assets",
            "candidate": key,
            "model": model_id,
            "prompt_count": len(presets),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "prompt_embeddings": str(embeddings_path),
            "prompt_embeddings_bytes": embeddings_path.stat().st_size,
            "uncond_embedding": str(uncond_embeddings_path),
            "uncond_embedding_bytes": uncond_embeddings_path.stat().st_size,
            "prompts": str(prompts_path),
            "scheduler": str(scheduler_path),
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
