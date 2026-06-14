#!/usr/bin/env python3
"""Low-memory wrapper around Apple's Stable Diffusion -> Core ML conversion flow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class ConversionStage:
    name: str
    extra_args: List[str]
    supports_quantization: bool = False


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stable Diffusion models to Core ML through Apple's "
            "ml-stable-diffusion package, using separate subprocesses per "
            "component to reduce RAM use."
        )
    )
    parser.add_argument("--apple-repo", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument(
        "--model-family",
        default="sdxl",
        choices=["sdxl", "sd15"],
        help="Conversion preset to use.",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Source model id from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--custom-vae-version",
        help="Optional custom VAE model id.",
    )
    parser.add_argument(
        "--latent-h",
        type=int,
        default=96,
        help="Latent height. 96=768px, 64=512px.",
    )
    parser.add_argument(
        "--latent-w",
        type=int,
        default=96,
        help="Latent width. 96=768px, 64=512px.",
    )
    parser.add_argument(
        "--attention-implementation",
        default="SPLIT_EINSUM",
        choices=["ORIGINAL", "SPLIT_EINSUM", "SPLIT_EINSUM_V2"],
    )
    parser.add_argument(
        "--quantize-nbits",
        type=int,
        choices=[2, 4, 6, 8],
    )
    parser.add_argument(
        "--quantize-text-encoder",
        action="store_true",
    )
    parser.add_argument(
        "--include-vae-encoder",
        action="store_true",
    )
    parser.add_argument(
        "--skip-chunk-unet",
        action="store_true",
    )
    parser.add_argument(
        "--disable-bundle",
        action="store_true",
    )
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def ensure_repo_layout(repo: Path) -> None:
    expected = repo / "python_coreml_stable_diffusion"
    if not expected.exists():
        raise SystemExit(
            f"Apple repo not found or unexpected layout: {repo}\n"
            "Expected to find python_coreml_stable_diffusion/ under that path."
        )


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


def base_command(args: argparse.Namespace, stage: ConversionStage) -> List[str]:
    command = [
        args.python_bin,
        "-m",
        "python_coreml_stable_diffusion.torch2coreml",
    ]
    if args.model_family == "sdxl":
        command.append("--xl-version")
    command.extend(
        [
            "--model-version",
            args.model_version,
            "--latent-h",
            str(args.latent_h),
            "--latent-w",
            str(args.latent_w),
            "--attention-implementation",
            args.attention_implementation,
            "-o",
            str(expand(args.output_dir)),
        ]
    )
    if args.custom_vae_version:
        command.extend(["--custom-vae-version", args.custom_vae_version])
    if args.quantize_nbits is not None and stage.supports_quantization:
        command.extend(["--quantize-nbits", str(args.quantize_nbits)])
    if not args.disable_bundle:
        command.append("--bundle-resources-for-swift-cli")
    if args.check_output_correctness:
        command.append("--check-output-correctness")
    return command


def build_stages(args: argparse.Namespace) -> Iterable[ConversionStage]:
    yield ConversionStage(
        "text_encoder",
        ["--convert-text-encoder"],
        supports_quantization=args.quantize_text_encoder,
    )
    yield ConversionStage("vae_decoder", ["--convert-vae-decoder"])
    yield ConversionStage("unet", ["--convert-unet"], supports_quantization=True)
    if args.include_vae_encoder:
        yield ConversionStage("vae_encoder", ["--convert-vae-encoder"])

    if not args.skip_chunk_unet:
        yield ConversionStage(
            "unet_chunk",
            ["--convert-unet", "--chunk-unet"],
            supports_quantization=True,
        )


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stage(
    repo: Path,
    stage: ConversionStage,
    command: List[str],
    dry_run: bool,
    env: dict[str, str],
) -> None:
    print(f"\n==> {stage.name}")
    print(shell_join(command))
    if dry_run:
        return

    subprocess.run(command, cwd=repo, check=True, env=env)


def main() -> int:
    args = parse_args()
    repo = expand(args.apple_repo)
    ensure_repo_layout(repo)

    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = expand(args.cache_dir)
    env = build_env(cache_dir)

    if args.model_family == "sdxl" and args.attention_implementation == "SPLIT_EINSUM_V2":
        print(
            "warning: Apple's SDXL guidance does not recommend SPLIT_EINSUM_V2 "
            "for mobile because compilation time can be prohibitively long.",
            file=sys.stderr,
        )

    for stage in build_stages(args):
        command = base_command(args, stage) + stage.extra_args
        run_stage(repo, stage, command, args.dry_run, env)

    resources_dir = output_dir / "Resources"
    if not args.disable_bundle:
        print("\nExpected Swift resources directory:")
        print(resources_dir)
    print("\nUsing cache directory:")
    print(cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
