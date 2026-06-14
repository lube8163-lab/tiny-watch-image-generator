from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "configs" / "model_candidates.json"


def load_candidates() -> dict[str, Any]:
    return json.loads(CANDIDATES_PATH.read_text())


def select_candidate(name: str | None) -> tuple[str, dict[str, Any]]:
    registry = load_candidates()
    key = name or registry["default"]
    candidates = registry["candidates"]
    if key not in candidates:
        known = ", ".join(sorted(candidates))
        raise SystemExit(f"unknown model candidate '{key}'. Known candidates: {known}")
    return key, candidates[key]


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["created_at_unix"] = int(time.time())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def require_diffusion_stack():
    missing = []
    modules = {}
    for name in ["torch", "diffusers", "PIL"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        deps = ", ".join(missing)
        raise SystemExit(
            f"missing dependencies: {deps}\n"
            "Create a research environment first:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        )
    return modules["torch"], modules["diffusers"]


def snapshot_download_candidate(candidate_key: str, candidate: dict[str, Any]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required:\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        ) from exc

    local_dir = ROOT / "models" / candidate_key
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    snapshot_download(
        repo_id=candidate["repo"],
        local_dir=local_dir,
        allow_patterns=candidate.get("download_allow_patterns"),
        ignore_patterns=candidate.get("download_ignore_patterns"),
    )
    return local_dir


def resolve_model_path(candidate_key: str, candidate: dict[str, Any], override: str | None, local_files_only: bool) -> str:
    if override:
        return override

    local_dir = ROOT / "models" / candidate_key
    if local_files_only:
        if not local_dir.exists():
            raise SystemExit(f"local model snapshot does not exist: {local_dir}")
        return str(local_dir)

    return str(snapshot_download_candidate(candidate_key, candidate))


def pick_torch_device(torch_module):
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def torch_dtype_for_device(torch_module, device) -> Any:
    if device.type in {"mps", "cuda"}:
        return torch_module.float16
    return torch_module.float32
