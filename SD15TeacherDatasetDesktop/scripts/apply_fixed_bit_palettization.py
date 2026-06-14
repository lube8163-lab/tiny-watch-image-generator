#!/usr/bin/env python3
"""Apply fixed-bit palettization to a converted Core ML model."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import coremltools as ct


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Palettize a converted Core ML mlpackage to a fixed bit width."
    )
    parser.add_argument("--mlpackage-path", required=True, help="Source .mlpackage path")
    parser.add_argument(
        "--output-mlpackage-path",
        required=True,
        help="Destination .mlpackage path for the palettized model",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        choices=[2, 4, 6, 8],
        required=True,
        help="Palette bit width",
    )
    parser.add_argument(
        "--compile-to",
        help="Optional output directory where the compiled .mlmodelc should be placed",
    )
    parser.add_argument(
        "--final-name",
        default="Unet",
        help="Final compiled model directory name without .mlmodelc",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    source_path = expand(args.mlpackage_path)
    output_path = expand(args.output_mlpackage_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.models.MLModel(str(source_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    op_config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=args.nbits)
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=op_config,
        op_type_configs={"gather": None},
    )

    palettized = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    palettized.save(str(output_path))
    print(f"Saved palettized model to {output_path}")

    if args.compile_to:
        compiled_path = compile_model(output_path, expand(args.compile_to), args.final_name)
        print(f"Compiled model to {compiled_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
