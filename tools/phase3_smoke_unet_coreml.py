#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from research_common import ROOT, write_manifest


def require_runtime():
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a fixed-shape Core ML UNet with random inputs.")
    parser.add_argument("model")
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=77)
    parser.add_argument("--cross-attention-dim", type=int, default=768)
    parser.add_argument("--time-cond-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    ct, np = require_runtime()
    rng = np.random.default_rng(args.seed)
    inputs = {
        "sample": rng.standard_normal((1, args.latent_channels, args.latent_height, args.latent_width)).astype(np.float16),
        "timestep": np.array([999.0], dtype=np.float16),
        "encoder_hidden_states": rng.standard_normal((1, args.prompt_length, args.cross_attention_dim)).astype(np.float16),
        "timestep_cond": rng.standard_normal((1, args.time_cond_dim)).astype(np.float16),
    }
    model = ct.models.MLModel(args.model)
    result = model.predict(inputs)
    noise_pred = result.get("noise_pred")
    if noise_pred is None:
        raise SystemExit(f"noise_pred output not found. Outputs: {', '.join(result.keys())}")

    manifest = Path(args.manifest) if args.manifest else ROOT / "reports" / "phase3" / "unet_smoke.json"
    write_manifest(
        manifest,
        {
            "phase": "phase3_smoke_unet_coreml",
            "model": args.model,
            "seed": args.seed,
            "noise_pred_shape": list(noise_pred.shape),
            "noise_pred_min": float(noise_pred.min()),
            "noise_pred_max": float(noise_pred.max()),
            "noise_pred_mean": float(noise_pred.mean()),
        },
    )
    print(manifest)


if __name__ == "__main__":
    main()
