#!/usr/bin/env python3
"""Convert official SwinIR lightweight x2 weights to Core ML for fixed iPhone sizes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import coremltools as ct
import torch


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def ensure_timm_compat() -> None:
    """Provide the tiny subset of timm used by SwinIR without adding a pip dependency."""
    layers = types.ModuleType("timm.models.layers")

    def to_2tuple(x):
        return (x, x) if not isinstance(x, tuple) else x

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float | None = None) -> None:
            super().__init__()
            self.drop_prob = drop_prob or 0.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = trunc_normal_

    sys.modules.setdefault("timm", types.ModuleType("timm"))
    sys.modules.setdefault("timm.models", types.ModuleType("timm.models"))
    sys.modules["timm.models.layers"] = layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SwinIR lightweight x2 to Core ML.")
    parser.add_argument("--swinir-repo", default=".build/SwinIR", help="Local clone of the official SwinIR repo")
    parser.add_argument(
        "--weights",
        default="artifacts/upscalers/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
        help="Official SwinIR lightweight x2 weight path",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/upscalers",
        help="Directory for generated mlpackage/mlmodelc outputs",
    )
    parser.add_argument(
        "--sizes",
        default="512,768",
        help="Comma-separated square input sizes to convert",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile each mlpackage to .mlmodelc",
    )
    return parser.parse_args()


def load_model(repo_path: Path, weights_path: Path) -> torch.nn.Module:
    ensure_timm_compat()
    sys.path.insert(0, str(repo_path))
    from models import network_swinir  # noqa: WPS433
    from models.network_swinir import SwinIR  # noqa: WPS433

    def static_batch_window_reverse(windows, window_size, height, width):
        x = windows.view(1, height // window_size, width // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(1, height, width, -1)
        return x

    network_swinir.window_reverse = static_batch_window_reverse

    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["params"] if "params" in state else state, strict=True)
    model.eval()
    return model


def compile_model(source_model_path: Path, output_dir: Path, final_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(source_model_path), str(output_dir)],
        check=True,
    )
    compiled_output = output_dir / f"{source_model_path.stem}.mlmodelc"
    final_path = output_dir / f"{final_name}.mlmodelc"
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.move(str(compiled_output), str(final_path))
    return final_path


def convert_size(model: torch.nn.Module, size: int, output_dir: Path, compile_to_mlmodelc: bool) -> None:
    sample = torch.rand(1, 3, size, size)
    traced = torch.jit.trace(model, sample)

    package_name = f"SuperResolution2x_{size}x{size}"
    output_path = output_dir / f"{package_name}.mlpackage"

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        inputs=[
            ct.ImageType(
                name="image",
                shape=sample.shape,
                scale=1 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.ImageType(
                name="output_image",
                color_layout=ct.colorlayout.RGB,
            )
        ],
    )
    mlmodel.save(str(output_path))
    print(f"Saved {output_path}")

    if compile_to_mlmodelc:
        compiled = compile_model(output_path, output_dir, package_name)
        print(f"Compiled {compiled}")


def main() -> int:
    args = parse_args()
    repo_path = expand(args.swinir_repo)
    weights_path = expand(args.weights)
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repo_path.joinpath("models/network_swinir.py").exists():
        raise SystemExit(f"SwinIR repo not found at {repo_path}")
    if not weights_path.exists():
        raise SystemExit(f"SwinIR weights not found at {weights_path}")

    model = load_model(repo_path, weights_path)
    sizes = [int(part.strip()) for part in args.sizes.split(",") if part.strip()]
    for size in sizes:
        if size % 8 != 0:
            raise SystemExit(f"Input size must be divisible by 8 for SwinIR windowing: {size}")
        convert_size(model, size, output_dir, compile_to_mlmodelc=args.compile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
