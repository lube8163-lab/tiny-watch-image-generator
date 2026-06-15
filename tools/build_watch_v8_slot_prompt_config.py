#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_watch_v4_prompt_config import DEFAULT_SOURCE, dedupe
from build_watch_v6_prompt_config import NEW_PRESETS, v6_preset
from build_watch_v7_focus_prompt_config import clean_preset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v8_slot_watch_freeprompt.json"


SLOT_FOCUS_KEYS = [
    "cat",
    "dog",
    "bird",
    "fish",
    "car",
    "flower",
    "house",
    "tree",
    "clock",
    "pizza",
    "apple",
    "astronaut",
    "alien",
    "dragon",
    "penguin",
    "frog",
]


SLOT_COLORS = ["red", "blue", "white", "black", "pink", "yellow", "green"]
SLOT_VIEWS = ["front view", "side view", "top view", "closeup"]
SLOT_STYLES = ["icon", "cartoon", "toy", "anime", "watercolor", "sketch", "photo"]
V8_NEGATIVE_TERMS = [
    "text",
    "logo",
    "watermark",
    "blurry",
    "noisy",
    "deformed",
    "cropped subject",
    "multiple subjects",
    "group",
    "duplicate subject",
    "collection",
    "collage",
    "patterned background",
    "busy background",
    "cluttered background",
    "scenery",
    "crowd",
    "grain",
    "speckles",
]
V8_GUARD_OVERRIDES = {
    "apple": "no logo, no bite",
    "car": "no road, no driver",
    "pizza": "one slice only",
}


ACTION_FALLBACKS = {
    "cat": ["sitting", "standing", "jumping", "sleeping"],
    "dog": ["running", "sitting", "standing", "sleeping"],
    "bird": ["flying", "standing", "looking"],
    "fish": ["swimming", "floating", "looking"],
    "car": ["parked", "tilted", "floating"],
    "flower": ["standing", "tilted", "shining"],
    "house": ["standing", "shining"],
    "tree": ["standing", "tilted"],
    "clock": ["standing", "tilted"],
    "pizza": ["standing", "tilted", "floating"],
    "apple": ["standing", "tilted", "floating"],
    "astronaut": ["standing", "floating", "walking"],
    "alien": ["standing", "walking", "smiling"],
    "dragon": ["standing", "flying", "sitting"],
    "penguin": ["standing", "walking", "sliding"],
    "frog": ["sitting", "jumping", "standing"],
}


def slot_preset(preset: dict[str, Any]) -> dict[str, Any]:
    output = clean_preset(preset)
    key = str(output["key"])
    if key in V8_GUARD_OVERRIDES:
        output["guard"] = V8_GUARD_OVERRIDES[key]
    output["colors_v2"] = dedupe(SLOT_COLORS + list(output.get("colors_v2") or []))[:7]
    output["views_v2"] = dedupe(SLOT_VIEWS + list(output.get("views_v2") or []))[:4]
    output["styles_v2"] = SLOT_STYLES
    output["actions_v2"] = dedupe(ACTION_FALLBACKS.get(key, []) + list(output.get("actions_v2") or []))[:5]
    output["modifiers_v2"] = dedupe(["simple", "small", "clear", "bright"] + list(output.get("modifiers_v2") or []))[:5]
    return output


def variants() -> list[dict[str, Any]]:
    clean_suffix = (
        "centered, whole subject visible, readable silhouette, "
        "plain light gray background, empty background"
    )
    strict_suffix = "single subject only, no repeated objects{guard_clause}, no scenery, no text"
    style_suffix = (
        "centered, whole subject visible, readable silhouette, plain light gray background, "
        "empty background, single subject only, no repeated objects, no text"
    )
    return [
        {
            "variant": "slot_base00",
            "conditioning_prompt": "{key}",
            "prompt": (
                "exactly one {subject}, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_base01",
            "conditioning_prompt": "simple {key}",
            "prompt": (
                "exactly one {subject}, simple icon, large clear shape, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_color_{color_token}",
            "slot_source": {"color": "colors_v2"},
            "max_expansions": 7,
            "conditioning_prompt": "{color} {key}",
            "prompt": (
                "exactly one {color} {subject}, clearly {color}, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_action_{action_token}",
            "slot_source": {"action": "actions_v2"},
            "max_expansions": 5,
            "conditioning_prompt": "{key} {action}",
            "prompt": (
                "exactly one {subject} {action}, simple readable pose, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_view_{view_token}",
            "slot_source": {"view": "views_v2"},
            "max_expansions": 4,
            "conditioning_prompt": "{key} {view}",
            "prompt": (
                "exactly one {subject}, {view}, clear viewpoint, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_style_simple_{style_token}",
            "slot_values": {"style": ["icon", "cartoon", "toy"]},
            "conditioning_prompt": "{style} {key}",
            "prompt": (
                "exactly one {subject}, {style} style, bold simple outline, smooth solid color areas, "
                f"{style_suffix}{{guard_clause}}"
            ),
        },
        {
            "variant": "slot_style_art_{style_token}",
            "slot_values": {"style": ["anime", "watercolor", "sketch"]},
            "conditioning_prompt": "{style} {key}",
            "prompt": (
                "exactly one {subject}, {style} style, simple clean illustration, minimal detail, "
                f"{style_suffix}{{guard_clause}}"
            ),
        },
        {
            "variant": "slot_style_photo",
            "slot_values": {"style": ["photo"]},
            "conditioning_prompt": "{style} {key}",
            "prompt": (
                "studio product photo of exactly one {subject}, simple soft lighting, "
                f"{style_suffix}{{guard_clause}}"
            ),
        },
        {
            "variant": "slot_color_action_{color_token}_{action_token}",
            "slot_source": {"color": "colors_v2", "action": "actions_v2"},
            "slot_strategy": "zip_cycle",
            "max_expansions": 7,
            "conditioning_prompt": "{color} {key} {action}",
            "prompt": (
                "exactly one {color} {subject} {action}, clearly {color}, simple readable pose, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_action_view_{action_token}_{view_token}",
            "slot_source": {"action": "actions_v2", "view": "views_v2"},
            "slot_strategy": "zip_cycle",
            "max_expansions": 8,
            "conditioning_prompt": "{key} {action} {view}",
            "prompt": (
                "exactly one {subject} {action}, {view}, clear pose and viewpoint, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_style_color_{style_token}_{color_token}",
            "slot_source": {"style": "styles_v2", "color": "colors_v2"},
            "slot_strategy": "zip_cycle",
            "max_expansions": 7,
            "conditioning_prompt": "{style} {color} {key}",
            "prompt": (
                "exactly one {color} {subject}, {style} style, clearly {color}, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
        {
            "variant": "slot_full_{style_token}_{color_token}_{action_token}_{view_token}",
            "slot_source": {
                "style": "styles_v2",
                "color": "colors_v2",
                "action": "actions_v2",
                "view": "views_v2",
            },
            "slot_strategy": "zip_cycle",
            "max_expansions": 14,
            "conditioning_prompt": "{style} {color} {key} {action} {view}",
            "prompt": (
                "exactly one {color} {subject} {action}, {view}, {style} style, "
                "clear color, pose, and viewpoint, "
                f"{clean_suffix}, {strict_suffix}"
            ),
        },
    ]


def count_variants(presets: list[dict[str, Any]]) -> int:
    total = 0
    for preset in presets:
        total += 2
        total += min(7, len(preset.get("colors_v2", [])))
        total += min(5, len(preset.get("actions_v2", [])))
        total += min(4, len(preset.get("views_v2", [])))
        total += 7
        total += min(7, len(preset.get("colors_v2", [])) * len(preset.get("actions_v2", [])))
        total += min(8, len(preset.get("actions_v2", [])) * len(preset.get("views_v2", [])))
        total += min(7, len(preset.get("styles_v2", [])) * len(preset.get("colors_v2", [])))
        total += min(
            14,
            len(preset.get("styles_v2", []))
            * len(preset.get("colors_v2", []))
            * len(preset.get("actions_v2", []))
            * len(preset.get("views_v2", [])),
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval-aligned v8 SDXL teacher prompts for watch slot repair.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    all_presets = [v6_preset(preset) for preset in source["presets"]]
    all_presets.extend(v6_preset(preset) for preset in NEW_PRESETS)
    by_key = {str(preset["key"]): preset for preset in all_presets}
    missing = [key for key in SLOT_FOCUS_KEYS if key not in by_key]
    if missing:
        raise SystemExit(f"missing v6 presets: {missing}")
    presets = [slot_preset(by_key[key]) for key in SLOT_FOCUS_KEYS]

    negative = ", ".join(dedupe(V8_NEGATIVE_TERMS))
    payload: dict[str, Any] = {
        "version": "watch75_sdxl_tiny_v8_slot_freeprompt",
        "description": (
            "Eval-aligned V8 teacher supplement for color/action/view/style slot binding. "
            "This is intentionally narrower than v7 and should be mixed with existing base/focus datasets."
        ),
        "default_negative_prompt": negative,
        "defaults": {
            "colors_v2": SLOT_COLORS,
            "views_v2": SLOT_VIEWS,
            "styles_v2": SLOT_STYLES,
            "modifiers_v2": ["simple", "small", "clear", "bright"],
        },
        "variants": variants(),
        "presets": presets,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"presets={len(presets)} variants_per_seed={count_variants(presets)}")


if __name__ == "__main__":
    main()
