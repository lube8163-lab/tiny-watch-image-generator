#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from research_common import ROOT, write_manifest


def require_runtime():
    try:
        import coremltools as ct
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "coremltools, numpy, and pillow are required:\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        ) from exc
    return ct, np


def chw_to_image(array, out: Path) -> None:
    # Core ML output is expected to be NCHW in roughly [-1, 1].
    image = array[0]
    image = ((image + 1.0) * 127.5).clip(0, 255).astype("uint8")
    image = image.transpose(1, 2, 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test an exported Core ML VAE decoder with random latents.")
    parser.add_argument("model", help="Path to VAE decoder .mlpackage")
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ct, np = require_runtime()
    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((1, args.latent_channels, args.latent_height, args.latent_width)).astype(np.float16)

    mlmodel = ct.models.MLModel(args.model)
    result = mlmodel.predict({"latents": latents})
    decoded = result.get("decoded")
    if decoded is None:
        known = ", ".join(result.keys())
        raise SystemExit(f"Core ML output 'decoded' not found. Outputs: {known}")

    out = Path(args.out) if args.out else ROOT / "reports" / "phase2" / "vae_decoder_smoke.png"
    chw_to_image(decoded, out)
    write_manifest(
        out.with_suffix(".json"),
        {
            "phase": "phase2_smoke_coreml",
            "model": args.model,
            "seed": args.seed,
            "latent_shape": list(latents.shape),
            "image": str(out),
        },
    )
    print(out)


if __name__ == "__main__":
    main()
