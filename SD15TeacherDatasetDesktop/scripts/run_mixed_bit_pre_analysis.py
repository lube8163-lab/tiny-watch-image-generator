#!/usr/bin/env python3
"""Run Apple's mixed-bit pre-analysis for SDXL UNet compression recipes."""

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
            "Run mixed_bit_compression_pre_analysis to generate candidate "
            "mixed-bit recipes for a model."
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
        "--model-version",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model id to analyze",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the analysis JSON",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/huggingface",
        help="Writable cache directory for Hugging Face and transformers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it",
    )
    return parser.parse_args()


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_env(cache_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = cache_dir / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HOME", str(cache_dir))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    env.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    return env


def main() -> int:
    args = parse_args()
    repo = expand(args.apple_repo)
    package_dir = repo / "python_coreml_stable_diffusion"
    if not package_dir.exists():
        raise SystemExit(
            f"Apple repo not found or unexpected layout: {repo}\n"
            "Expected to find python_coreml_stable_diffusion/ under that path."
        )

    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = expand(args.cache_dir)
    env = build_env(cache_dir)

    command = [
        args.python_bin,
        "-m",
        "python_coreml_stable_diffusion.mixed_bit_compression_pre_analysis",
        "--model-version",
        args.model_version,
        "-o",
        str(output_dir),
    ]

    print(shell_join(command))
    if args.dry_run:
        return 0

    subprocess.run(command, cwd=repo, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
