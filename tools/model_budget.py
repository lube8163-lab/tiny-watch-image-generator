#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate parameter storage for tiny txt2img targets.")
    parser.add_argument("--params", type=float, default=350, help="Parameter count in millions")
    args = parser.parse_args()

    params = args.params * 1_000_000
    rows = [
        ("fp16", 16),
        ("int8", 8),
        ("int6", 6),
        ("int4", 4),
        ("int3", 3),
    ]
    for name, bits in rows:
        mib = params * bits / 8 / 1024 / 1024
        print(f"{name:>4}: {mib:7.1f} MiB for weights only")

    print("\nRuntime memory will be higher because activations, decoded tensors, and Core ML planning buffers are not included.")


if __name__ == "__main__":
    main()
