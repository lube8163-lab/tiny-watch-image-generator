#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_watch_v4_prompt_config import DEFAULT_SOURCE, as_list, build_preset, dedupe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v5_watch_freeprompt.json"

STYLE_DEFAULTS_V5 = ["icon", "cartoon", "toy", "sketch", "watercolor"]
VIEW_DEFAULTS_V5 = ["front view", "side view", "closeup"]

ACTION_OVERRIDES = {
    "apple": ["standing", "floating", "tilted"],
    "banana": ["floating", "tilted"],
    "orange": ["standing", "floating", "tilted"],
    "strawberry": ["standing", "floating", "tilted"],
    "bread": ["standing", "floating", "tilted"],
    "pizza": ["standing", "floating", "tilted"],
    "cake": ["standing", "floating"],
    "bird": ["flying", "standing", "looking"],
    "butterfly": ["flying", "floating"],
    "fish": ["swimming", "jumping", "turning"],
    "boat": ["floating", "parked"],
    "airplane": ["flying", "tilted"],
    "rocket": ["flying", "standing", "tilted"],
    "car": ["parked", "tilted"],
    "bus": ["parked", "standing"],
    "train": ["parked", "standing"],
    "bicycle": ["standing", "tilted"],
    "cat": ["sitting", "standing", "jumping", "sleeping", "looking"],
    "dog": ["sitting", "standing", "running", "jumping", "looking"],
    "rabbit": ["sitting", "standing", "jumping", "looking"],
    "bear": ["sitting", "standing", "looking"],
    "horse": ["standing", "running", "jumping"],
    "fox": ["sitting", "standing", "jumping", "looking"],
    "owl": ["sitting", "standing", "looking"],
    "robot": ["standing", "walking", "holding", "looking"],
    "face": ["smiling", "looking", "standing"],
    "star": ["shining", "floating", "tilted"],
    "sun": ["shining", "floating"],
    "moon": ["floating", "shining"],
    "cloud": ["floating", "drifting"],
    "ball": ["rolling", "bouncing", "floating"],
    "tree": ["standing", "leaning", "swaying"],
    "flower": ["standing", "tilted"],
    "mushroom": ["standing", "tilted"],
    "clock": ["standing", "tilted"],
    "guitar": ["standing", "tilted"],
    "shoe": ["standing", "tilted"],
    "chair": ["standing", "tilted"],
}

MODIFIER_OVERRIDES = {
    "cat": ["cute", "small", "fluffy", "simple"],
    "dog": ["cute", "small", "fluffy", "simple"],
    "rabbit": ["cute", "small", "fluffy", "simple"],
    "bear": ["cute", "round", "fluffy", "small"],
    "fox": ["cute", "small", "fluffy", "simple"],
    "owl": ["cute", "small", "round", "simple"],
    "face": ["round", "simple", "bright", "cute"],
    "star": ["bright", "small", "large", "simple"],
    "sun": ["bright", "round", "large", "simple"],
    "moon": ["bright", "round", "small", "simple"],
    "cloud": ["fluffy", "soft", "small", "simple"],
    "ball": ["round", "small", "large", "bright"],
    "tree": ["small", "large", "simple", "dark"],
    "flower": ["small", "bright", "cute", "simple"],
    "bread": ["small", "round", "simple", "bright"],
    "pizza": ["small", "simple", "bright"],
    "orange": ["round", "bright", "small"],
}

STYLE_OVERRIDES = {
    "cat": ["cartoon", "toy", "sketch", "anime", "watercolor"],
    "dog": ["cartoon", "toy", "sketch", "anime", "watercolor"],
    "rabbit": ["cartoon", "toy", "sketch", "anime", "watercolor"],
    "face": ["cartoon", "sketch", "anime", "watercolor"],
    "flower": ["icon", "cartoon", "sketch", "watercolor"],
    "tree": ["icon", "cartoon", "sketch", "watercolor"],
    "cloud": ["icon", "cartoon", "sketch", "watercolor"],
    "star": ["icon", "cartoon", "toy", "sketch"],
    "sun": ["icon", "cartoon", "toy", "sketch"],
    "moon": ["icon", "cartoon", "toy", "sketch"],
}

VIEW_OVERRIDES = {
    "book": ["front view", "side view", "top view", "closeup"],
    "clock": ["front view", "side view", "closeup"],
    "ball": ["front view", "top view", "closeup"],
    "pizza": ["front view", "side view", "closeup"],
    "bread": ["front view", "side view", "closeup"],
    "face": ["front view", "side view", "closeup"],
    "car": ["front view", "side view", "top view"],
    "bus": ["front view", "side view"],
    "train": ["front view", "side view"],
    "bicycle": ["side view", "front view", "closeup"],
}

COLOR_EXTRAS = {
    "face": ["brown", "gray"],
    "tree": ["green", "brown"],
    "cloud": ["white", "blue", "gray"],
    "clock": ["white", "black", "gray"],
    "ball": ["red", "blue", "yellow"],
    "heart": ["red", "pink", "white"],
    "cake": ["pink", "white", "brown"],
    "pizza": ["yellow", "orange"],
    "bread": ["brown", "golden"],
}


def v5_preset(preset: dict[str, Any]) -> dict[str, Any]:
    output = build_preset(preset)
    key = str(output["key"])
    output["colors_v2"] = dedupe(COLOR_EXTRAS.get(key, []) + as_list(output.get("colors_v2")))[:4]
    output["actions_v2"] = dedupe(ACTION_OVERRIDES.get(key, []) + as_list(output.get("actions_v2")))[:5]
    output["modifiers_v2"] = dedupe(MODIFIER_OVERRIDES.get(key, []) + as_list(output.get("modifiers_v2")))[:4]
    output["styles_v2"] = dedupe(STYLE_OVERRIDES.get(key, STYLE_DEFAULTS_V5))[:5]
    output["views_v2"] = dedupe(VIEW_OVERRIDES.get(key, VIEW_DEFAULTS_V5))[:4]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v5 watch teacher prompt config.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    presets = [v5_preset(preset) for preset in source["presets"]]
    payload: dict[str, Any] = {
        "version": "watch46_sdxl_tiny_v5_freeprompt",
        "description": (
            "V5 adds broader action, modifier, view, and style coverage for the compositional_v2 "
            "watch model while keeping single-subject plain-background teacher images."
        ),
        "default_negative_prompt": source["default_negative_prompt"],
        "defaults": {
            "views_v2": VIEW_DEFAULTS_V5,
            "styles_v2": STYLE_DEFAULTS_V5,
        },
        "variants": [
            {
                "variant": "v00",
                "conditioning_prompt": "{key}",
                "prompt": "exactly one {subject}, isolated single subject, centered, full subject visible, clean simple illustration, matte plain light gray background, empty background, readable silhouette, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "clear00",
                "conditioning_prompt": "simple {key}",
                "prompt": "exactly one {subject}, simple clean object icon, centered, bold readable silhouette, low detail, plain light gray background, empty background, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "shape00",
                "conditioning_prompt": "icon {key}",
                "prompt": "exactly one {subject}, icon style, clear outline, high contrast readable shape, centered, full subject visible, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "view{view_index}",
                "slot_source": {"view": "views_v2"},
                "max_expansions": 4,
                "conditioning_prompt": "{key} {view}",
                "prompt": "exactly one {subject}, {view}, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "act{action_index}",
                "slot_source": {"action": "actions_v2"},
                "max_expansions": 5,
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
                "max_expansions": 4,
                "conditioning_prompt": "{modifier} {key}",
                "prompt": "exactly one {modifier} {subject}, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "style{style_index}",
                "slot_source": {"style": "styles_v2"},
                "max_expansions": 5,
                "conditioning_prompt": "{style} {key}",
                "prompt": "exactly one {subject}, {style} style, isolated single subject, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "combo{color_index}_{action_index}",
                "slot_source": {"color": "colors_v2", "action": "actions_v2"},
                "max_expansions": 8,
                "conditioning_prompt": "{color} {key} {action}",
                "prompt": "exactly one {color} {subject} {action}, isolated single subject, centered, full subject visible, simple readable pose, clean illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "aview{action_index}_{view_index}",
                "slot_source": {"action": "actions_v2", "view": "views_v2"},
                "max_expansions": 6,
                "conditioning_prompt": "{key} {action} {view}",
                "prompt": "exactly one {subject} {action}, {view}, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "scolor{style_index}_{color_index}",
                "slot_source": {"style": "styles_v2", "color": "colors_v2"},
                "max_expansions": 6,
                "conditioning_prompt": "{style} {color} {key}",
                "prompt": "exactly one {color} {subject}, {style} style, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
            {
                "variant": "mact{modifier_index}_{action_index}",
                "slot_source": {"modifier": "modifiers_v2", "action": "actions_v2"},
                "max_expansions": 6,
                "conditioning_prompt": "{modifier} {key} {action}",
                "prompt": "exactly one {modifier} {subject} {action}, centered, full subject visible, simple readable pose, clean illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
            },
        ],
        "presets": presets,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    variant_count = 0
    for preset in presets:
        variant_count += 3
        variant_count += min(4, len(preset["views_v2"]))
        variant_count += min(5, len(preset["actions_v2"]))
        variant_count += min(4, len(preset["colors_v2"]))
        variant_count += min(4, len(preset["modifiers_v2"]))
        variant_count += min(5, len(preset["styles_v2"]))
        variant_count += min(8, len(preset["colors_v2"]) * len(preset["actions_v2"]))
        variant_count += min(6, len(preset["actions_v2"]) * len(preset["views_v2"]))
        variant_count += min(6, len(preset["styles_v2"]) * len(preset["colors_v2"]))
        variant_count += min(6, len(preset["modifiers_v2"]) * len(preset["actions_v2"]))
    print(f"wrote {args.out}")
    print(f"presets={len(presets)} variants_per_seed={variant_count}")


if __name__ == "__main__":
    main()
