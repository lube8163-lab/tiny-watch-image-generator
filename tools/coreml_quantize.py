#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_coremltools():
    try:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
    except ImportError as exc:
        raise SystemExit(
            "coremltools is not installed. Install it in a separate env on macOS:\n"
            "  python3 -m pip install coremltools\n"
        ) from exc
    return ct, cto


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress an existing Core ML component for the Apple Watch txt2img path."
    )
    parser.add_argument("input", help="Input .mlpackage or .mlmodel")
    parser.add_argument("--out", required=True, help="Output .mlpackage")
    parser.add_argument(
        "--mode",
        choices=["linear8", "palettize4", "palettize6"],
        default="palettize4",
    )
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    ct, cto = require_coremltools()
    model = ct.models.MLModel(args.input)

    if args.mode == "linear8":
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        compressed = cto.linear_quantize_weights(model, config)
    else:
        nbits = 4 if args.mode == "palettize4" else 6
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=nbits)
        )
        compressed = cto.palettize_weights(model, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(out)

    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".json")
    manifest = {
        "input": str(Path(args.input)),
        "output": str(out),
        "mode": args.mode,
        "output_bytes": directory_size(out) if out.is_dir() else out.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    main()
