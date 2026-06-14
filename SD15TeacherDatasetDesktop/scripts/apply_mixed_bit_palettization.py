#!/usr/bin/env python3
"""Apply a mixed-bit palettization recipe to a converted Core ML UNet."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an existing mixed-bit recipe using Apple's "
            "mixed_bit_compression_apply.py helper."
        )
    )
    parser.add_argument(
        "--apple-repo",
        required=True,
        help="Path to a local clone of github.com/apple/ml-stable-diffusion",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to run Apple's helper script",
    )
    parser.add_argument(
        "--converted-model-dir",
        required=True,
        help="Directory produced by scripts/convert_sdxl.py",
    )
    parser.add_argument(
        "--pre-analysis-json",
        required=True,
        help="Path to the JSON produced by mixed_bit_compression_pre_analysis",
    )
    parser.add_argument(
        "--selected-recipe",
        required=True,
        help='Recipe key such as "recipe_4.04_bit_mixedpalette"',
    )
    parser.add_argument(
        "--unet-name",
        default="Unet.mlpackage",
        help="UNet package name inside the converted model directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it",
    )
    return parser.parse_args()


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    args = parse_args()

    repo = expand(args.apple_repo)
    script_path = repo / "python_coreml_stable_diffusion" / "mixed_bit_compression_apply.py"
    if not script_path.exists():
        raise SystemExit(
            f"Could not find Apple's mixed-bit apply helper at {script_path}"
        )

    converted_model_dir = expand(args.converted_model_dir)
    pre_analysis_json = expand(args.pre_analysis_json)
    output_dir = converted_model_dir / "palettized"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_bin,
        "-m",
        "python_coreml_stable_diffusion.mixed_bit_compression_apply",
        "--mlpackage-path",
        str(converted_model_dir / args.unet_name),
        "-o",
        str(output_dir),
        "--pre-analysis-json-path",
        str(pre_analysis_json),
        "--selected-recipe",
        args.selected_recipe,
    ]

    print(shell_join(command))
    if args.dry_run:
        return 0

    subprocess.run(command, cwd=repo, check=True)
    print(f"\nPalettized output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
