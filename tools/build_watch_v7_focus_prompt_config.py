#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_watch_v4_prompt_config import DEFAULT_SOURCE, dedupe
from build_watch_v6_prompt_config import NEW_PRESETS, v6_preset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v7_focus_watch_freeprompt.json"


FOCUS_KEYS = [
    "cat",
    "dog",
    "rabbit",
    "fox",
    "owl",
    "fish",
    "bird",
    "penguin",
    "duck",
    "frog",
    "monkey",
    "dragon",
    "alien",
    "cloud",
    "tree",
    "flower",
    "heart",
    "star",
    "ball",
    "fire",
    "diamond",
    "crown",
    "train",
    "car",
    "bicycle",
    "airplane",
    "boat",
    "umbrella",
    "key",
    "bottle",
    "pencil",
    "phone",
    "computer",
    "sword",
    "shield",
    "apple",
    "banana",
    "orange",
    "strawberry",
    "bread",
    "pizza",
    "cake",
    "icecream",
    "donut",
    "sushi",
]


SUBJECT_OVERRIDES = {
    "heart": "single heart symbol with a clean bold outline",
    "star": "single five pointed star symbol with a clean bold outline",
    "ball": "single round ball with one simple shadow",
    "fire": "single simple flame icon",
    "cloud": "single simple fluffy cloud shape",
    "train": "single toy train engine side silhouette",
    "car": "single small toy car side silhouette",
    "bicycle": "single simple bicycle side silhouette",
    "airplane": "single simple airplane silhouette",
    "boat": "single small simple boat silhouette",
    "key": "single simple metal key icon",
    "bottle": "single simple bottle icon",
    "pencil": "single simple pencil icon",
    "phone": "single simple smartphone icon",
    "computer": "single open laptop computer icon",
    "sword": "single simple sword icon",
    "shield": "single simple shield icon",
    "apple": "single apple fruit icon with a stem",
    "banana": "single banana fruit icon",
    "orange": "single orange fruit icon",
    "strawberry": "single strawberry fruit icon",
    "bread": "single bread loaf icon",
    "pizza": "single pizza slice icon",
    "cake": "single cake slice icon",
    "icecream": "single ice cream cone with one scoop",
    "donut": "single donut icon",
    "sushi": "single sushi piece icon",
}


GUARD_OVERRIDES = {
    "apple": "fruit only, no apple logo, no bite mark, no repeated fruit",
    "banana": "fruit only, no bunch, no repeated bananas",
    "orange": "fruit only, one orange only, no orange slices, no repeated fruit, no pattern",
    "strawberry": "fruit only, one strawberry only, no repeated berries",
    "bread": "one bread loaf only, no basket, no plate, no multiple loaves, no repeated pattern",
    "pizza": "one pizza slice only, no full pizza, no plate, no multiple slices, no repeated toppings pattern",
    "cake": "one cake slice only, no candles, no plate, no text, no multiple desserts",
    "icecream": "one ice cream cone only, no cup, no spoon, no text, no repeated desserts",
    "donut": "one donut only, no plate, no multiple donuts, no sprinkles pattern background",
    "sushi": "one sushi piece only, no plate, no chopsticks, no multiple sushi pieces",
    "cloud": "one cloud only, no sky scene, no rain, no repeated clouds, no weather background",
    "fire": "one flame only, no campfire, no candle, no smoke, no repeated flames",
    "train": "one train engine only, no tracks, no station, no scenery, no multiple cars",
    "car": "one car only, no road, no driver, no buildings, no scenery",
    "bicycle": "one bicycle only, no rider, no street, no scenery",
    "airplane": "one airplane only, no airport, no clouds, no sky scene",
    "boat": "one boat only, no ocean scene, no waves, no people",
    "umbrella": "one umbrella only, no rain scene, no person, no repeated umbrellas",
    "key": "one key only, no keychain, no lock, no repeated keys",
    "bottle": "one bottle only, no label text, no table, no repeated bottles",
    "pencil": "one pencil only, no paper, no text, no repeated pencils",
    "phone": "one phone only, blank screen, no text, no app icons, no hand",
    "computer": "one laptop only, blank screen, no text, no desk, no hand",
    "sword": "one sword only, no person, no shield, no background scene",
    "shield": "one shield only, no person, no sword, no emblem text",
}


CLEAN_STYLES = ["icon", "cartoon", "toy", "sketch"]


def clean_preset(preset: dict[str, Any]) -> dict[str, Any]:
    output = dict(v6_preset(preset))
    key = str(output["key"])
    if key in SUBJECT_OVERRIDES:
        output["subject"] = SUBJECT_OVERRIDES[key]
    output["guard"] = GUARD_OVERRIDES.get(key, output.get("guard"))
    output["styles_v2"] = CLEAN_STYLES
    output["modifiers_v2"] = dedupe(
        ["simple", "small", "round", "bright"] + list(output.get("modifiers_v2") or [])
    )[:5]
    output["views_v2"] = dedupe(["front view", "side view", "closeup", "top view"] + list(output.get("views_v2") or []))[:4]
    return output


def variants() -> list[dict[str, Any]]:
    clean_suffix = (
        "flat vector style, smooth solid fills, minimal shading, clean readable silhouette, "
        "plain matte light gray background, empty background, no texture, no grain, no paper texture"
    )
    icon_suffix = (
        "simple icon style, bold outline, smooth solid color areas, high contrast readable shape, "
        "plain light gray background, empty background, no texture, no pattern"
    )
    return [
        {
            "variant": "clean00",
            "conditioning_prompt": "{key}",
            "prompt": (
                "exactly one {subject}, isolated single subject, centered, full subject visible, "
                f"{clean_suffix}, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "clean01",
            "conditioning_prompt": "simple {key}",
            "prompt": (
                "exactly one {subject}, very simple rounded object icon, centered, large clear silhouette, "
                f"{clean_suffix}, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "icon00",
            "conditioning_prompt": "icon {key}",
            "prompt": (
                "exactly one {subject}, centered object icon, full subject visible, "
                f"{icon_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "cartoon00",
            "conditioning_prompt": "cartoon {key}",
            "prompt": (
                "exactly one {subject}, simple cartoon icon, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "toy00",
            "conditioning_prompt": "toy {key}",
            "prompt": (
                "exactly one {subject}, small toy-like object, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "view{view_index}",
            "slot_source": {"view": "views_v2"},
            "max_expansions": 4,
            "conditioning_prompt": "{key} {view}",
            "prompt": (
                "exactly one {subject}, {view}, isolated single subject, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "color{color_index}",
            "slot_source": {"color": "colors_v2"},
            "max_expansions": 4,
            "conditioning_prompt": "{color} {key}",
            "prompt": (
                "exactly one {color} {subject}, isolated single subject, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "act{action_index}",
            "slot_source": {"action": "actions_v2"},
            "max_expansions": 5,
            "conditioning_prompt": "{key} {action}",
            "prompt": (
                "exactly one {subject} {action}, simple readable pose, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "mod{modifier_index}",
            "slot_source": {"modifier": "modifiers_v2"},
            "max_expansions": 5,
            "conditioning_prompt": "{modifier} {key}",
            "prompt": (
                "exactly one {modifier} {subject}, isolated single subject, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "cview{color_index}_{view_index}",
            "slot_source": {"color": "colors_v2", "view": "views_v2"},
            "max_expansions": 6,
            "conditioning_prompt": "{color} {key} {view}",
            "prompt": (
                "exactly one {color} {subject}, {view}, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "aview{action_index}_{view_index}",
            "slot_source": {"action": "actions_v2", "view": "views_v2"},
            "max_expansions": 6,
            "conditioning_prompt": "{key} {action} {view}",
            "prompt": (
                "exactly one {subject} {action}, {view}, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "mview{modifier_index}_{view_index}",
            "slot_source": {"modifier": "modifiers_v2", "view": "views_v2"},
            "max_expansions": 6,
            "conditioning_prompt": "{modifier} {key} {view}",
            "prompt": (
                "exactly one {modifier} {subject}, {view}, centered, full subject visible, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
        {
            "variant": "cact{color_index}_{action_index}",
            "slot_source": {"color": "colors_v2", "action": "actions_v2"},
            "max_expansions": 6,
            "conditioning_prompt": "{color} {key} {action}",
            "prompt": (
                "exactly one {color} {subject} {action}, centered, full subject visible, simple readable pose, "
                f"{clean_suffix}, no repeated objects, not a collection{{guard_clause}}, no text, no logo, no watermark"
            ),
        },
    ]


def count_variants(presets: list[dict[str, Any]]) -> int:
    total = 0
    for preset in presets:
        total += 5
        total += min(4, len(preset.get("views_v2", [])))
        total += min(4, len(preset.get("colors_v2", [])))
        total += min(5, len(preset.get("actions_v2", [])))
        total += min(5, len(preset.get("modifiers_v2", [])))
        total += min(6, len(preset.get("colors_v2", [])) * len(preset.get("views_v2", [])))
        total += min(6, len(preset.get("actions_v2", [])) * len(preset.get("views_v2", [])))
        total += min(6, len(preset.get("modifiers_v2", [])) * len(preset.get("views_v2", [])))
        total += min(6, len(preset.get("colors_v2", [])) * len(preset.get("actions_v2", [])))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused v7 SDXL teacher prompts for watch quality repair.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    all_presets = [v6_preset(preset) for preset in source["presets"]]
    all_presets.extend(v6_preset(preset) for preset in NEW_PRESETS)
    by_key = {str(preset["key"]): preset for preset in all_presets}
    missing = [key for key in FOCUS_KEYS if key not in by_key]
    if missing:
        raise SystemExit(f"missing v6 presets: {missing}")
    presets = [clean_preset(by_key[key]) for key in FOCUS_KEYS]

    negative = ", ".join(
        dedupe(
            [
                source["default_negative_prompt"],
                "grain",
                "speckles",
                "mottled texture",
                "paper texture",
                "canvas texture",
                "dirty background",
                "noisy background",
                "background pattern",
                "tiny details",
                "photorealistic clutter",
            ]
        )
    )
    payload: dict[str, Any] = {
        "version": "watch75_sdxl_tiny_v7_focus_freeprompt",
        "description": (
            "Focused V7 teacher supplement for the v6 watch model. It keeps the existing vocabulary "
            "but targets food, clean symbols, tools, vehicles, and action/view pairs with lower-texture "
            "single-subject SDXL prompts."
        ),
        "default_negative_prompt": negative,
        "defaults": {
            "views_v2": ["front view", "side view", "closeup", "top view"],
            "styles_v2": CLEAN_STYLES,
            "modifiers_v2": ["simple", "small", "round", "bright"],
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
