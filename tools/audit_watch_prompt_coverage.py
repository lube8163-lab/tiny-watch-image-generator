#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prompt_normalization as prompt_norm  # noqa: E402


def load_eval_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    prompts: list[dict[str, Any]] = []
    for group in payload.get("groups", []):
        group_key = str(group.get("key") or "")
        for index, prompt in enumerate(group.get("prompts", [])):
            prompts.append({"group": group_key, "index": index, "prompt": str(prompt)})
    return prompts


def content_view_prompt_keys(path: Path) -> set[str]:
    text = path.read_text()
    return {
        key
        for key in re.findall(r"\.init\(key: \"([^\"]+)\", title:", text)
        if key
    }


def slot_payload(prompt: str) -> dict[str, Any]:
    slots = prompt_norm.normalize_prompt_slots(prompt, include_views=True)
    return {
        "canonical": prompt_norm.canonicalize_prompt(prompt),
        "subjects": list(slots.subjects),
        "colors": list(slots.colors),
        "actions": list(slots.actions),
        "views": list(slots.views),
        "modifiers": list(slots.modifiers),
        "styles": list(slots.styles),
        "unknownTokens": list(slots.unknown_tokens),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    eval_prompts = load_eval_prompts(args.eval_config)
    prompt_entries: list[dict[str, Any]] = []
    without_subject: list[dict[str, Any]] = []
    with_unknowns: list[dict[str, Any]] = []

    for item in eval_prompts:
        slots = slot_payload(item["prompt"])
        entry = {**item, **slots}
        prompt_entries.append(entry)
        if not slots["subjects"]:
            without_subject.append(entry)
        if slots["unknownTokens"]:
            with_unknowns.append(entry)

    ui_keys = content_view_prompt_keys(args.content_view)
    generator_subjects = set(prompt_norm.PROMPT_ALIASES)
    slot_keys = (
        set(prompt_norm.COLOR_ALIASES)
        | set(prompt_norm.ACTION_ALIASES)
        | set(prompt_norm.STYLE_ALIASES)
        | {"front view", "side view", "back view", "top view", "closeup"}
    )

    return {
        "evalConfig": str(args.eval_config),
        "contentView": str(args.content_view),
        "summary": {
            "promptCount": len(prompt_entries),
            "withoutSubjectCount": len(without_subject),
            "withUnknownTokensCount": len(with_unknowns),
            "generatorSubjectCount": len(generator_subjects),
            "uiPromptKeyCount": len(ui_keys),
        },
        "missingUiSubjects": sorted(generator_subjects - ui_keys),
        "extraUiPromptKeys": sorted(ui_keys - generator_subjects - slot_keys),
        "withoutSubject": without_subject,
        "withUnknownTokens": with_unknowns,
        "prompts": prompt_entries if args.include_prompts else [],
    }


def print_human(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("watch prompt coverage")
    print(f"  prompts: {summary['promptCount']}")
    print(f"  prompts without subject slot: {summary['withoutSubjectCount']}")
    print(f"  prompts with unknown tokens: {summary['withUnknownTokensCount']}")
    print(f"  generator subjects: {summary['generatorSubjectCount']}")
    print(f"  UI prompt keys: {summary['uiPromptKeyCount']}")

    if report["missingUiSubjects"]:
        print("\nmissing UI subjects:")
        for key in report["missingUiSubjects"]:
            print(f"  - {key}")

    if report["extraUiPromptKeys"]:
        print("\nextra UI prompt keys:")
        for key in report["extraUiPromptKeys"]:
            print(f"  - {key}")

    if report["withoutSubject"]:
        print("\nprompts without subject slot:")
        for entry in report["withoutSubject"]:
            print(f"  - {entry['group']}[{entry['index']}]: {entry['prompt']} -> {entry['canonical']}")

    if report["withUnknownTokens"]:
        print("\nprompts with unknown tokens:")
        for entry in report["withUnknownTokens"]:
            unknown = ", ".join(entry["unknownTokens"])
            print(f"  - {entry['group']}[{entry['index']}]: {entry['prompt']} -> {unknown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit watch prompt slot and UI coverage without GPU work.")
    parser.add_argument("--eval-config", type=Path, default=ROOT / "configs" / "prompt_eval_suite.json")
    parser.add_argument(
        "--content-view",
        type=Path,
        default=ROOT / "watchos_example" / "TinyImageWatchApp" / "ContentView.swift",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--include-prompts", action="store_true", help="Include every prompt entry in JSON output.")
    parser.add_argument("--fail-on-missing-ui", action="store_true")
    parser.add_argument("--fail-on-unknown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human(report)

    should_fail = False
    if args.fail_on_missing_ui and report["missingUiSubjects"]:
        should_fail = True
    if args.fail_on_unknown and report["withUnknownTokens"]:
        should_fail = True
    if should_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
