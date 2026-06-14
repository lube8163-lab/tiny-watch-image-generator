#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SCHEME_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    component: str
    quantization: str
    size_bytes: int
    size_mb: float
    sidecar_json: str | None
    sidecar: dict[str, Any] | None
    watch_disk_tier: str
    watch_note: str


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def classify_component(path: Path) -> str:
    name = path.name.lower()
    if "unet" in name:
        return "denoiser_unet"
    if "vae_decoder" in name or "decoder" in name:
        return "vae_decoder"
    if "text_encoder" in name:
        return "text_encoder"
    if "controlnet" in name:
        return "controlnet"
    if "safety" in name:
        return "safety_checker"
    return "unknown"


def classify_quantization(path: Path, sidecar: dict[str, Any] | None) -> str:
    name = path.name.lower()
    if sidecar and isinstance(sidecar.get("mode"), str):
        return sidecar["mode"]
    for token in ("2bit", "3bit", "4bit", "6bit", "8bit"):
        if token in name:
            return token
    if "palett" in name:
        return "palettized"
    return "fp16_or_unlabeled"


def read_sidecar(path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidate = path.with_suffix(".json")
    if not candidate.exists():
        return None, None
    try:
        return candidate, json.loads(candidate.read_text())
    except json.JSONDecodeError:
        return candidate, {"_error": "invalid_json"}


def disk_tier(size_mb: float, component: str) -> tuple[str, str]:
    if size_mb <= 25:
        return "small", "reasonable first Watch probe, still profile runtime memory"
    if size_mb <= 75:
        return "medium", "possible only for isolated component tests"
    if size_mb <= 160:
        if component == "denoiser_unet":
            return "large", "direct Watch denoiser test is high risk but worth a 1-step probe"
        return "large", "large for Watch; test load before prediction"
    if size_mb <= 450:
        return "very_large", "likely teacher/iPhone/Mac asset unless Watch load surprisingly succeeds"
    return "extreme", "not a direct Watch candidate"


def iter_artifacts(scan_roots: Iterable[Path]) -> Iterable[Path]:
    for root in scan_roots:
        if not root.exists():
            continue
        yield from root.rglob("*.mlpackage")
        yield from root.rglob("*.mlmodelc")


def build_report(scan_roots: list[Path]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for path in sorted(set(iter_artifacts(scan_roots))):
        sidecar_path, sidecar = read_sidecar(path)
        size_bytes = directory_size(path)
        size_mb = size_bytes / 1024 / 1024
        component = classify_component(path)
        quantization = classify_quantization(path, sidecar)
        tier, note = disk_tier(size_mb, component)
        artifacts.append(
            Artifact(
                path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                kind=path.suffix.lstrip("."),
                component=component,
                quantization=quantization,
                size_bytes=size_bytes,
                size_mb=round(size_mb, 2),
                sidecar_json=str(sidecar_path.relative_to(ROOT)) if sidecar_path and sidecar_path.is_relative_to(ROOT) else str(sidecar_path) if sidecar_path else None,
                sidecar=sidecar,
                watch_disk_tier=tier,
                watch_note=note,
            )
        )
    return artifacts


def print_table(artifacts: list[Artifact]) -> None:
    if not artifacts:
        print("No Core ML artifacts found.")
        return

    rows = sorted(artifacts, key=lambda item: (item.size_mb, item.path))
    print(f"{'size':>9}  {'tier':<10}  {'component':<14}  {'quant':<18}  path")
    print("-" * 110)
    for item in rows:
        print(
            f"{item.size_mb:8.2f}M  "
            f"{item.watch_disk_tier:<10}  "
            f"{item.component:<14}  "
            f"{item.quantization:<18}  "
            f"{item.path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Core ML artifacts for the Apple Watch SD quantization scheme."
    )
    parser.add_argument(
        "--scan",
        action="append",
        default=None,
        help="Directory to scan. Can be passed multiple times. Defaults to dist/ and this scheme's artifacts/.",
    )
    parser.add_argument(
        "--out",
        default=str(SCHEME_ROOT / "reports" / "coreml_artifact_report.json"),
        help="JSON report path. Keep this under schemes/watch_sd_quantization/reports/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_roots = [ROOT / "dist", SCHEME_ROOT / "artifacts"] if args.scan is None else [Path(item).expanduser().resolve() for item in args.scan]
    artifacts = build_report(scan_roots)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scheme": "watch_sd_quantization",
        "repo_root": str(ROOT),
        "scan_roots": [str(path) for path in scan_roots],
        "artifact_count": len(artifacts),
        "artifacts": [asdict(item) for item in sorted(artifacts, key=lambda item: item.size_bytes)],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print_table(artifacts)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
