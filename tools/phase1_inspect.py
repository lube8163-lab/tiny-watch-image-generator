#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from research_common import (
    ROOT,
    require_diffusion_stack,
    resolve_model_path,
    select_candidate,
    write_manifest,
)


def count_parameters(module) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def parameter_bytes(module) -> int:
    if module is None:
        return 0
    return sum(param.numel() * param.element_size() for param in module.parameters())


def component_report(pipe) -> dict:
    components = {}
    for name in ["text_encoder", "text_encoder_2", "unet", "transformer", "vae"]:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        components[name] = {
            "parameters": count_parameters(module),
            "parameter_bytes": parameter_bytes(module),
        }
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: inspect candidate model component sizes.")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    model_id = resolve_model_path(key, candidate, args.model, args.local_files_only)
    out = Path(args.out) if args.out else ROOT / "reports" / "phase1" / key / "components.json"

    torch, diffusers = require_diffusion_stack()
    pipeline_cls = getattr(diffusers, candidate["pipeline"], None)
    if pipeline_cls is None:
        raise SystemExit(f"diffusers does not expose {candidate['pipeline']}; update diffusers.")

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    report = {
        "phase": "phase1_inspect",
        "candidate": key,
        "model": model_id,
        "source_repo": candidate["repo"],
        "components": component_report(pipe),
    }
    write_manifest(out, report)
    print(out)


if __name__ == "__main__":
    main()
