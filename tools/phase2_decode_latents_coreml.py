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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode saved float16 latents with a Core ML VAE decoder.")
    parser.add_argument("model")
    parser.add_argument("latents")
    parser.add_argument("--latent-height", type=int, default=32)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ct, np = require_runtime()
    shape = (1, args.latent_channels, args.latent_height, args.latent_width)
    latents = np.fromfile(args.latents, dtype=np.float16).reshape(shape)
    model = ct.models.MLModel(args.model)
    result = model.predict({"latents": latents})
    decoded = result["decoded"][0]
    image = ((decoded + 1.0) * 127.5).clip(0, 255).astype("uint8").transpose(1, 2, 0)

    out = Path(args.out) if args.out else ROOT / "reports" / "phase2" / "decoded_latents_coreml.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    write_manifest(
        out.with_suffix(".json"),
        {
            "phase": "phase2_decode_latents_coreml",
            "model": args.model,
            "latents": args.latents,
            "latent_shape": list(shape),
            "image": str(out),
        },
    )
    print(out)


if __name__ == "__main__":
    main()
