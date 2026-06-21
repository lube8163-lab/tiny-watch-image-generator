#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_common import ROOT, directory_size, resolve_model_path, select_candidate, write_manifest


DEFAULT_PROMPTS = ["cat mascot", "white mascot", "cat logo"]


def load_stack():
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name in ["coremltools", "torch", "transformers"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "missing dependencies: "
            + ", ".join(missing)
            + "\nRun with the repo venv, for example: ./.venv/bin/python"
        )
    return modules["coremltools"], modules["torch"], modules["transformers"]


class CLIPTextEncoderWrapper:
    def __init__(self, torch_module, text_encoder):
        self.torch = torch_module
        self.text_encoder = text_encoder

    def module(self):
        torch = self.torch
        text_encoder = self.text_encoder

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = text_encoder

            def forward(self, input_ids):
                outputs = self.text_encoder(input_ids=input_ids.to(torch.long))
                return outputs.last_hidden_state

        return Wrapped().eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export/probe the LCM CLIP text encoder as a fixed-shape Core ML component."
    )
    parser.add_argument("--candidate", default="lcm_dreamshaper_v7")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--convert", action="store_true", help="Also convert the fixed-shape text encoder to Core ML.")
    parser.add_argument("--quantize-nbits", type=int, choices=[4, 6, 8], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ct, torch, transformers = load_stack()
    key, candidate = select_candidate(args.candidate)
    model_root = Path(resolve_model_path(key, candidate, args.model, args.local_files_only))
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "dist" / key / "text_encoder_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        model_root / "tokenizer",
        local_files_only=True,
    )
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        model_root / "text_encoder",
        local_files_only=True,
    ).eval()

    tokenized = tokenizer(
        args.prompts,
        padding="max_length",
        truncation=True,
        max_length=int(text_encoder.config.max_position_embeddings),
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(torch.int32)
    with torch.inference_mode():
        reference = text_encoder(input_ids=tokenized.input_ids).last_hidden_state

    prompts_path = out_dir / "text_encoder_probe_prompts.json"
    ids_path = out_dir / "input_ids_i32.bin"
    reference_path = out_dir / "reference_hidden_states_f16.bin"
    prompts_path.write_text(
        json.dumps(
            {
                "prompts": args.prompts,
                "inputIdsShape": list(input_ids.shape),
                "inputIdsDtype": "int32",
                "hiddenStatesShape": list(reference.shape),
                "hiddenStatesDtype": "float16",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    input_ids.numpy().astype(np.int32, copy=False).tofile(ids_path)
    reference.detach().cpu().numpy().astype(np.float16, copy=False).tofile(reference_path)

    package_path: Path | None = None
    package_size = 0
    if args.convert:
        wrapper = CLIPTextEncoderWrapper(torch, text_encoder).module()
        example = input_ids[0:1]
        traced = torch.jit.trace(wrapper, example, strict=False)
        package_path = out_dir / "clip_text_encoder_77.mlpackage"
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.watchOS10,
            inputs=[ct.TensorType(name="input_ids", shape=example.shape, dtype=np.int32)],
            outputs=[ct.TensorType(name="hidden_states")],
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(package_path)
        package_size = directory_size(package_path)

        if args.quantize_nbits is not None:
            try:
                from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
            except ImportError as exc:
                raise SystemExit("coremltools optimize API is unavailable in this environment") from exc
            quantized_path = out_dir / f"clip_text_encoder_77_{args.quantize_nbits}bit.mlpackage"
            config = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    mode="kmeans",
                    nbits=args.quantize_nbits,
                )
            )
            quantized = palettize_weights(mlmodel, config)
            quantized.save(quantized_path)
            package_path = quantized_path
            package_size = directory_size(quantized_path)

    manifest = {
        "phase": "export_clip_text_encoder_coreml",
        "candidate": key,
        "model": str(model_root),
        "parameter_count": sum(param.numel() for param in text_encoder.parameters()),
        "parameter_bytes_fp32": sum(param.numel() * param.element_size() for param in text_encoder.parameters()),
        "max_position_embeddings": int(text_encoder.config.max_position_embeddings),
        "hidden_size": int(text_encoder.config.hidden_size),
        "vocab_size": int(text_encoder.config.vocab_size),
        "prompts": args.prompts,
        "prompt_asset": str(prompts_path),
        "input_ids": str(ids_path),
        "reference_hidden_states": str(reference_path),
        "coreml_package": str(package_path) if package_path else None,
        "coreml_package_bytes": package_size,
        "converted": package_path is not None,
        "quantize_nbits": args.quantize_nbits,
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
