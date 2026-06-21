#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path


def require_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit(
            "coremltools is required:\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        ) from exc
    return ct


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def chunk_outputs(output_dir: Path) -> tuple[Path, Path]:
    outputs = sorted(output_dir.glob("*_chunk1.mlpackage")) + sorted(output_dir.glob("chunk1.mlpackage"))
    chunk1 = outputs[0] if outputs else None
    outputs = sorted(output_dir.glob("*_chunk2.mlpackage")) + sorted(output_dir.glob("chunk2.mlpackage"))
    chunk2 = outputs[0] if outputs else None
    if chunk1 is None or chunk2 is None:
        found = ", ".join(path.name for path in output_dir.glob("*.mlpackage"))
        raise SystemExit(f"bisect output missing chunk1/chunk2 in {output_dir}; found: {found}")
    return chunk1, chunk2


def split_to_parts(input_model: Path, output_dir: Path, parts: int) -> list[Path]:
    ct = require_coremltools()
    levels = int(math.log2(parts))
    current = [input_model]
    for level in range(levels):
        next_level: list[Path] = []
        for index, model in enumerate(current):
            split_dir = output_dir / "splits" / f"level{level + 1}_{index + 1}"
            split_dir.mkdir(parents=True, exist_ok=False)
            print(f"split level={level + 1}/{levels} model={model} -> {split_dir}", flush=True)
            ct.models.utils.bisect_model(
                str(model),
                str(split_dir),
                merge_chunks_to_pipeline=False,
                check_output_correctness=False,
            )
            chunk1, chunk2 = chunk_outputs(split_dir)
            next_level.extend([chunk1, chunk2])
        current = next_level
    return current


def compile_parts(parts: list[Path], compiled_dir: Path, prefix: str) -> list[Path]:
    compiled_dir.mkdir(parents=True, exist_ok=True)
    compiled_parts: list[Path] = []
    for index, part in enumerate(parts, start=1):
        target = compiled_dir / f"{prefix}_part{index}.mlmodelc"
        if target.exists():
            raise SystemExit(f"compiled target already exists: {target}")
        print(f"compile part={index}/{len(parts)} {part.name} -> {target.name}", flush=True)
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(part), str(compiled_dir)],
            check=True,
        )
        produced = compiled_dir / f"{part.stem}.mlmodelc"
        if not produced.exists():
            found = ", ".join(path.name for path in compiled_dir.glob("*.mlmodelc"))
            raise SystemExit(f"coremlcompiler output missing {produced}; found: {found}")
        shutil.move(str(produced), str(target))
        compiled_parts.append(target)
    return compiled_parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively bisect a Core ML package and optionally compile named parts.")
    parser.add_argument("input", help="Input .mlpackage")
    parser.add_argument("--parts", type=int, default=8, help="Number of chunks; must be a power of two.")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--compile-out-dir", default=None)
    parser.add_argument("--compile-prefix", default=None)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    input_model = Path(args.input)
    if not input_model.exists():
        raise SystemExit(f"input model does not exist: {input_model}")
    if not power_of_two(args.parts):
        raise SystemExit("--parts must be a power of two")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else Path("/private/tmp") / f"{input_model.stem}_chunks_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=False)
    parts = split_to_parts(input_model, work_dir, args.parts)
    print("parts:")
    for index, part in enumerate(parts, start=1):
        print(f"  {index}: {part} {directory_size(part)} bytes")

    if args.skip_compile:
        return

    if args.compile_out_dir is None:
        raise SystemExit("--compile-out-dir is required unless --skip-compile is set")
    prefix = args.compile_prefix or input_model.stem
    compiled = compile_parts(parts, Path(args.compile_out_dir), prefix)
    print("compiled:")
    for index, part in enumerate(compiled, start=1):
        print(f"  {index}: {part} {directory_size(part)} bytes")


if __name__ == "__main__":
    main()
