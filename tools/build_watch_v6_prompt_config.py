#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_watch_v4_prompt_config import DEFAULT_SOURCE, as_list, dedupe
from build_watch_v5_prompt_config import (
    ACTION_OVERRIDES,
    COLOR_EXTRAS,
    MODIFIER_OVERRIDES,
    STYLE_DEFAULTS_V5,
    STYLE_OVERRIDES,
    VIEW_DEFAULTS_V5,
    VIEW_OVERRIDES,
    v5_preset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "configs" / "sdxl_tiny_teacher_prompts_v6_watch_freeprompt.json"


NEW_PRESETS: list[dict[str, Any]] = [
    {
        "key": "astronaut",
        "title": "Astronaut",
        "subject": "single cute astronaut in a round helmet",
        "modifiers_v2": ["small", "round", "white", "simple"],
        "colors_v2": ["white", "silver", "blue"],
        "actions_v2": ["standing", "floating", "walking", "looking"],
    },
    {
        "key": "alien",
        "title": "Alien",
        "subject": "single cute alien creature with a big head",
        "modifiers_v2": ["cute", "small", "round", "simple"],
        "colors_v2": ["green", "blue", "purple"],
        "actions_v2": ["standing", "walking", "looking", "smiling"],
    },
    {
        "key": "dragon",
        "title": "Dragon",
        "subject": "single cute dragon with small wings",
        "modifiers_v2": ["cute", "small", "fluffy", "simple"],
        "colors_v2": ["green", "red", "blue"],
        "actions_v2": ["standing", "flying", "sitting", "looking"],
    },
    {
        "key": "penguin",
        "title": "Penguin",
        "subject": "single cute penguin",
        "modifiers_v2": ["cute", "small", "round", "simple"],
        "colors_v2": ["black", "white", "blue"],
        "actions_v2": ["standing", "walking", "sliding", "looking"],
    },
    {
        "key": "turtle",
        "title": "Turtle",
        "subject": "single turtle with round shell",
        "modifiers_v2": ["small", "round", "simple", "cute"],
        "colors_v2": ["green", "brown", "yellow"],
        "actions_v2": ["standing", "walking", "swimming", "looking"],
    },
    {
        "key": "elephant",
        "title": "Elephant",
        "subject": "single cute elephant",
        "modifiers_v2": ["large", "round", "cute", "simple"],
        "colors_v2": ["gray", "blue", "white"],
        "actions_v2": ["standing", "walking", "sitting", "looking"],
    },
    {
        "key": "lion",
        "title": "Lion",
        "subject": "single lion with round mane",
        "modifiers_v2": ["cute", "round", "fluffy", "simple"],
        "colors_v2": ["yellow", "orange", "brown"],
        "actions_v2": ["standing", "sitting", "walking", "looking"],
    },
    {
        "key": "monkey",
        "title": "Monkey",
        "subject": "single cute monkey",
        "modifiers_v2": ["cute", "small", "round", "simple"],
        "colors_v2": ["brown", "orange", "gray"],
        "actions_v2": ["sitting", "standing", "jumping", "looking"],
    },
    {
        "key": "frog",
        "title": "Frog",
        "subject": "single cute frog",
        "modifiers_v2": ["cute", "small", "round", "simple"],
        "colors_v2": ["green", "yellow", "blue"],
        "actions_v2": ["sitting", "jumping", "standing", "looking"],
    },
    {
        "key": "duck",
        "title": "Duck",
        "subject": "single cute duck",
        "modifiers_v2": ["cute", "small", "round", "simple"],
        "colors_v2": ["yellow", "white", "brown"],
        "actions_v2": ["standing", "swimming", "walking", "looking"],
    },
    {
        "key": "deer",
        "title": "Deer",
        "subject": "single deer with antlers",
        "modifiers_v2": ["cute", "small", "simple", "brown"],
        "colors_v2": ["brown", "white", "gray"],
        "actions_v2": ["standing", "walking", "running", "looking"],
    },
    {
        "key": "whale",
        "title": "Whale",
        "subject": "single blue whale",
        "modifiers_v2": ["large", "round", "simple", "cute"],
        "colors_v2": ["blue", "gray", "white"],
        "actions_v2": ["swimming", "floating", "jumping"],
    },
    {
        "key": "umbrella",
        "title": "Umbrella",
        "subject": "single open umbrella",
        "modifiers_v2": ["open", "round", "small", "simple"],
        "colors_v2": ["red", "blue", "yellow", "black"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
    {
        "key": "key",
        "title": "Key",
        "subject": "single metal key",
        "modifiers_v2": ["metallic", "small", "simple", "bright"],
        "colors_v2": ["golden", "silver", "gray"],
        "actions_v2": ["floating", "tilted", "standing"],
    },
    {
        "key": "bottle",
        "title": "Bottle",
        "subject": "single bottle",
        "modifiers_v2": ["small", "simple", "transparent", "round"],
        "colors_v2": ["blue", "green", "white", "brown"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
    {
        "key": "pencil",
        "title": "Pencil",
        "subject": "single pencil",
        "modifiers_v2": ["small", "simple", "sharp", "wooden"],
        "colors_v2": ["yellow", "red", "blue"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
    {
        "key": "lamp",
        "title": "Lamp",
        "subject": "single table lamp",
        "modifiers_v2": ["bright", "small", "simple", "round"],
        "colors_v2": ["yellow", "white", "black"],
        "actions_v2": ["standing", "shining", "tilted"],
    },
    {
        "key": "phone",
        "title": "Phone",
        "subject": "single smartphone",
        "modifiers_v2": ["small", "simple", "black", "bright"],
        "colors_v2": ["black", "white", "blue"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
    {
        "key": "computer",
        "title": "Computer",
        "subject": "single laptop computer",
        "modifiers_v2": ["open", "small", "simple", "gray"],
        "colors_v2": ["gray", "black", "silver"],
        "actions_v2": ["standing", "open", "tilted"],
    },
    {
        "key": "crown",
        "title": "Crown",
        "subject": "single crown",
        "modifiers_v2": ["golden", "bright", "simple", "small"],
        "colors_v2": ["golden", "yellow", "silver"],
        "actions_v2": ["standing", "floating", "shining"],
    },
    {
        "key": "diamond",
        "title": "Diamond",
        "subject": "single diamond gem",
        "modifiers_v2": ["bright", "transparent", "simple", "shiny"],
        "colors_v2": ["blue", "white", "purple"],
        "actions_v2": ["shining", "floating", "tilted"],
    },
    {
        "key": "sword",
        "title": "Sword",
        "subject": "single sword",
        "modifiers_v2": ["metallic", "sharp", "simple", "bright"],
        "colors_v2": ["silver", "gray", "blue"],
        "actions_v2": ["standing", "tilted", "shining"],
    },
    {
        "key": "shield",
        "title": "Shield",
        "subject": "single shield",
        "modifiers_v2": ["round", "metallic", "simple", "bright"],
        "colors_v2": ["red", "blue", "silver"],
        "actions_v2": ["standing", "tilted", "shining"],
    },
    {
        "key": "cactus",
        "title": "Cactus",
        "subject": "single cactus plant",
        "modifiers_v2": ["small", "green", "simple", "spiky"],
        "colors_v2": ["green", "brown", "yellow"],
        "actions_v2": ["standing", "tilted"],
    },
    {
        "key": "volcano",
        "title": "Volcano",
        "subject": "single volcano mountain",
        "modifiers_v2": ["large", "simple", "dark", "bright"],
        "colors_v2": ["brown", "gray", "orange"],
        "actions_v2": ["standing", "shining"],
    },
    {
        "key": "fire",
        "title": "Fire",
        "subject": "single flame fire icon",
        "modifiers_v2": ["bright", "small", "simple", "shining"],
        "colors_v2": ["orange", "red", "yellow"],
        "actions_v2": ["shining", "floating", "standing"],
    },
    {
        "key": "icecream",
        "title": "Ice Cream",
        "subject": "single ice cream cone",
        "modifiers_v2": ["small", "round", "cute", "simple"],
        "colors_v2": ["pink", "white", "brown", "yellow"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
    {
        "key": "donut",
        "title": "Donut",
        "subject": "single donut",
        "modifiers_v2": ["round", "small", "cute", "simple"],
        "colors_v2": ["brown", "pink", "white"],
        "actions_v2": ["standing", "floating", "rolling"],
    },
    {
        "key": "sushi",
        "title": "Sushi",
        "subject": "single sushi piece",
        "modifiers_v2": ["small", "round", "simple", "white"],
        "colors_v2": ["white", "orange", "black"],
        "actions_v2": ["standing", "tilted", "floating"],
    },
]

V6_MODIFIERS = [
    "striped",
    "spotted",
    "open",
    "closed",
    "broken",
    "wooden",
    "metallic",
    "transparent",
    "sharp",
    "spiky",
]

V6_ACTIONS = ["dancing", "climbing", "spinning", "sliding", "open"]

V6_ACTION_OVERRIDES = {
    "astronaut": ["floating", "standing", "walking", "looking"],
    "alien": ["standing", "walking", "looking", "smiling"],
    "dragon": ["standing", "flying", "sitting", "looking"],
    "penguin": ["standing", "walking", "sliding", "looking"],
    "turtle": ["walking", "swimming", "standing", "looking"],
    "elephant": ["standing", "walking", "sitting", "looking"],
    "lion": ["standing", "sitting", "walking", "looking"],
    "monkey": ["sitting", "jumping", "climbing", "looking"],
    "frog": ["sitting", "jumping", "standing", "looking"],
    "duck": ["standing", "swimming", "walking", "looking"],
    "deer": ["standing", "walking", "running", "looking"],
    "whale": ["swimming", "floating", "jumping"],
    "umbrella": ["standing", "tilted", "floating"],
    "lamp": ["standing", "shining", "tilted"],
    "diamond": ["shining", "floating", "spinning"],
    "fire": ["shining", "floating", "spinning"],
}

V6_MODIFIER_OVERRIDES = {
    "dragon": ["cute", "small", "spiky", "simple"],
    "penguin": ["cute", "small", "round", "simple"],
    "turtle": ["small", "round", "spotted", "simple"],
    "lion": ["cute", "round", "fluffy", "simple"],
    "monkey": ["cute", "small", "brown", "simple"],
    "frog": ["cute", "small", "round", "spotted"],
    "duck": ["cute", "small", "round", "simple"],
    "deer": ["small", "brown", "spotted", "simple"],
    "umbrella": ["open", "striped", "round", "simple"],
    "key": ["metallic", "small", "bright", "simple"],
    "pencil": ["wooden", "sharp", "small", "simple"],
    "computer": ["open", "small", "gray", "simple"],
    "diamond": ["transparent", "bright", "sharp", "simple"],
    "cactus": ["green", "spiky", "small", "simple"],
}

V6_VIEW_OVERRIDES = {
    "umbrella": ["front view", "side view", "top view"],
    "key": ["front view", "side view", "closeup"],
    "bottle": ["front view", "side view", "closeup"],
    "pencil": ["side view", "front view", "closeup"],
    "phone": ["front view", "side view", "closeup"],
    "computer": ["front view", "side view", "top view"],
    "diamond": ["front view", "side view", "closeup"],
    "sword": ["front view", "side view", "tilted"],
    "shield": ["front view", "side view", "closeup"],
}


def v6_preset(preset: dict[str, Any]) -> dict[str, Any]:
    output = v5_preset(preset)
    key = str(output["key"])
    output["colors_v2"] = dedupe(
        as_list(preset.get("colors_v2")) + COLOR_EXTRAS.get(key, []) + as_list(output.get("colors_v2"))
    )[:4]
    output["actions_v2"] = dedupe(
        V6_ACTION_OVERRIDES.get(key, [])
        + ACTION_OVERRIDES.get(key, [])
        + as_list(preset.get("actions_v2"))
        + as_list(output.get("actions_v2"))
    )[:5]
    output["modifiers_v2"] = dedupe(
        V6_MODIFIER_OVERRIDES.get(key, [])
        + MODIFIER_OVERRIDES.get(key, [])
        + as_list(preset.get("modifiers_v2"))
        + as_list(output.get("modifiers_v2"))
        + V6_MODIFIERS[:2]
    )[:5]
    output["styles_v2"] = dedupe(STYLE_OVERRIDES.get(key, STYLE_DEFAULTS_V5))[:5]
    output["views_v2"] = dedupe(
        V6_VIEW_OVERRIDES.get(key, []) + VIEW_OVERRIDES.get(key, VIEW_DEFAULTS_V5)
    )[:4]
    return output


def variants() -> list[dict[str, Any]]:
    return [
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
            "max_expansions": 5,
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
            "variant": "mview{modifier_index}_{view_index}",
            "slot_source": {"modifier": "modifiers_v2", "view": "views_v2"},
            "max_expansions": 6,
            "conditioning_prompt": "{modifier} {key} {view}",
            "prompt": "exactly one {modifier} {subject}, {view}, centered, full subject visible, clean simple illustration, plain light gray background, empty background, no repeated objects, not a collection{guard_clause}, no text, no logo, no watermark",
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
    ]


def count_variants(presets: list[dict[str, Any]]) -> int:
    total = 0
    for preset in presets:
        total += 3
        total += min(4, len(preset["views_v2"]))
        total += min(5, len(preset["actions_v2"]))
        total += min(4, len(preset["colors_v2"]))
        total += min(5, len(preset["modifiers_v2"]))
        total += min(5, len(preset["styles_v2"]))
        total += min(8, len(preset["colors_v2"]) * len(preset["actions_v2"]))
        total += min(6, len(preset["actions_v2"]) * len(preset["views_v2"]))
        total += min(6, len(preset["modifiers_v2"]) * len(preset["views_v2"]))
        total += min(6, len(preset["styles_v2"]) * len(preset["colors_v2"]))
        total += min(6, len(preset["modifiers_v2"]) * len(preset["actions_v2"]))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v6 watch teacher prompt config.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    presets = [v6_preset(preset) for preset in source["presets"]]
    presets.extend(v6_preset(preset) for preset in NEW_PRESETS)
    payload: dict[str, Any] = {
        "version": "watch75_sdxl_tiny_v6_freeprompt",
        "description": (
            "V6 keeps the v5 watch core categories and adds compact new subjects plus broader "
            "action, modifier, material/state, and view coverage for constrained free prompts."
        ),
        "default_negative_prompt": source["default_negative_prompt"],
        "defaults": {
            "views_v2": VIEW_DEFAULTS_V5,
            "styles_v2": STYLE_DEFAULTS_V5,
            "modifiers_v2": V6_MODIFIERS,
            "actions_v2": V6_ACTIONS,
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
