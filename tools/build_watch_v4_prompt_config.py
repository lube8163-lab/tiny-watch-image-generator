#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v3_watch.json"
DEFAULT_OUT = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v4_watch_freeprompt.json"


KNOWN_COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "golden",
    "silver",
]

COLOR_DEFAULTS = {
    "apple": ["red", "green"],
    "banana": ["yellow"],
    "orange": ["orange"],
    "strawberry": ["red", "pink"],
    "star": ["golden", "blue"],
    "sun": ["yellow", "golden"],
    "moon": ["silver", "soft blue"],
    "bird": ["blue", "yellow"],
    "fish": ["orange", "blue"],
    "flower": ["pink", "yellow"],
    "car": ["red", "blue"],
    "bus": ["yellow", "blue"],
    "train": ["red", "blue"],
    "robot": ["silver", "blue"],
    "cat": ["gray", "white"],
    "dog": ["brown", "white"],
    "rabbit": ["white", "gray"],
    "bear": ["brown"],
    "horse": ["brown", "white"],
    "airplane": ["white", "red"],
    "boat": ["blue", "brown"],
    "bicycle": ["red", "black"],
    "rocket": ["red", "silver"],
}

ACTION_DEFAULTS = {
    "bird": ["flying", "standing"],
    "butterfly": ["flying"],
    "fish": ["swimming", "jumping"],
    "boat": ["parked"],
    "airplane": ["flying"],
    "rocket": ["flying", "standing"],
    "car": ["parked"],
    "bus": ["parked"],
    "train": ["parked"],
    "bicycle": ["standing"],
    "cat": ["sitting", "standing", "jumping", "sleeping"],
    "dog": ["sitting", "standing", "running", "jumping"],
    "rabbit": ["sitting", "standing", "jumping"],
    "bear": ["sitting", "standing"],
    "horse": ["standing", "running", "jumping"],
    "robot": ["standing", "walking", "holding"],
    "face": ["standing"],
}

MODIFIER_DEFAULTS = {
    "cat": ["cute", "small"],
    "dog": ["cute", "small"],
    "rabbit": ["cute", "small"],
    "bear": ["cute", "round"],
    "flower": ["small", "bright"],
    "star": ["bright", "small"],
    "sun": ["bright", "round"],
    "moon": ["round", "bright"],
    "ball": ["round", "small"],
    "heart": ["round", "bright"],
}

STYLE_DEFAULTS = ["icon", "cartoon", "toy", "sketch"]
VIEW_DEFAULTS = ["front view", "side view", "closeup"]


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(item.lower().replace("-", " ").split())
        if key and key not in seen:
            seen.add(key)
            output.append(item)
    return output


def extract_colors(preset: dict[str, Any]) -> list[str]:
    text = " ".join(as_list(preset.get("modifiers")) + as_list(preset.get("subject"))).lower()
    colors = [color for color in KNOWN_COLORS if color in text]
    colors.extend(COLOR_DEFAULTS.get(str(preset["key"]), []))
    return dedupe(colors)[:4]


def build_preset(preset: dict[str, Any]) -> dict[str, Any]:
    key = str(preset["key"])
    output = {
        "key": key,
        "title": str(preset.get("title") or key.title()),
        "subject": str(preset.get("subject") or key),
    }
    for optional_key in ["guard", "qc_flags"]:
        if optional_key in preset:
            output[optional_key] = preset[optional_key]
    output["colors_v2"] = extract_colors(preset)
    output["actions_v2"] = dedupe(ACTION_DEFAULTS.get(key, ["standing"]))[:4]
    output["modifiers_v2"] = dedupe(MODIFIER_DEFAULTS.get(key, ["small", "simple"]))[:3]
    output["styles_v2"] = STYLE_DEFAULTS
    output["views_v2"] = VIEW_DEFAULTS
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v2-encoder-aligned watch teacher prompt config.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    presets = [build_preset(preset) for preset in source["presets"]]
    payload = {
        "version": "watch46_sdxl_tiny_v4_freeprompt",
        "description": (
            "Watch teacher prompts aligned to compositional_v2. The schedule keeps plain backgrounds, "
            "uses explicit subject/color/action/view/style slots, and avoids unrecognized modifier wording."
        ),
        "default_negative_prompt": source["default_negative_prompt"],
        "defaults": {
            "views_v2": VIEW_DEFAULTS,
            "styles_v2": STYLE_DEFAULTS,
        },
        "variants": [
            {
                "variant": "v00",
                "conditioning_prompt": "{key}",
                "prompt": "exactly one {subject}, isolated single subject, centered, full subject visible, clean simple illustration, matte plain light gray background, empty background, readable silhouette, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "v01",
                "conditioning_prompt": "simple {key}",
                "prompt": "exactly one {subject}, simple object icon, centered, bold silhouette, low detail readable shape, clean vector-like illustration, plain light gray background, empty background, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "view{view_index}",
                "slot_source": {"view": "views_v2"},
                "max_expansions": 3,
                "conditioning_prompt": "{key} {view}",
                "prompt": "exactly one {subject}, {view}, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "act{action_index}",
                "slot_source": {"action": "actions_v2"},
                "max_expansions": 4,
                "conditioning_prompt": "{key} {action}",
                "prompt": "exactly one {subject} {action}, isolated single subject, centered, full subject visible, simple readable pose, clean illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "color{color_index}",
                "slot_source": {"color": "colors_v2"},
                "max_expansions": 4,
                "conditioning_prompt": "{color} {key}",
                "prompt": "exactly one {color} {subject}, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "mod{modifier_index}",
                "slot_source": {"modifier": "modifiers_v2"},
                "max_expansions": 3,
                "conditioning_prompt": "{modifier} {key}",
                "prompt": "exactly one {modifier} {subject}, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "style{style_index}",
                "slot_source": {"style": "styles_v2"},
                "max_expansions": 4,
                "conditioning_prompt": "{style} {key}",
                "prompt": "exactly one {subject}, {style} style, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "combo{color_index}_{action_index}",
                "slot_source": {"color": "colors_v2", "action": "actions_v2"},
                "max_expansions": 6,
                "conditioning_prompt": "{color} {key} {action}",
                "prompt": "exactly one {color} {subject} {action}, isolated single subject, centered, full subject visible, simple readable pose, clean illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
        ],
        "presets": presets,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    variant_count = 0
    for preset in presets:
        variant_count += 2
        variant_count += min(3, len(preset["views_v2"]))
        variant_count += min(4, len(preset["actions_v2"]))
        variant_count += min(4, len(preset["colors_v2"]))
        variant_count += min(3, len(preset["modifiers_v2"]))
        variant_count += min(4, len(preset["styles_v2"]))
        variant_count += min(6, len(preset["colors_v2"]) * len(preset["actions_v2"]))
    print(f"wrote {args.out}")
    print(f"presets={len(presets)} variants_per_seed={variant_count}")


if __name__ == "__main__":
    main()
