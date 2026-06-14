from __future__ import annotations

import re
from dataclasses import dataclass

MASK_64 = 0xFFFFFFFFFFFFFFFF
FNV_OFFSET_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3

PROMPT_ENCODER_HASH = "hash_v1"
PROMPT_ENCODER_COMPOSITIONAL = "compositional_v1"
PROMPT_ENCODER_COMPOSITIONAL_V2 = "compositional_v2"
PROMPT_ENCODERS = (PROMPT_ENCODER_HASH, PROMPT_ENCODER_COMPOSITIONAL, PROMPT_ENCODER_COMPOSITIONAL_V2)


PROMPT_ALIASES: dict[str, tuple[str, ...]] = {
    "astronaut": ("astronaut", "spaceperson", "spaceman", "宇宙飛行士"),
    "alien": ("alien", "aliens", "extraterrestrial", "宇宙人"),
    "dragon": ("dragon", "dragons", "竜", "ドラゴン"),
    "penguin": ("penguin", "penguins", "ペンギン"),
    "turtle": ("turtle", "turtles", "亀", "カメ"),
    "elephant": ("elephant", "elephants", "象", "ゾウ"),
    "lion": ("lion", "lions", "ライオン"),
    "monkey": ("monkey", "monkeys", "猿", "サル"),
    "frog": ("frog", "frogs", "蛙", "カエル"),
    "duck": ("duck", "ducks", "アヒル", "鴨"),
    "deer": ("deer", "鹿", "シカ"),
    "whale": ("whale", "whales", "くじら", "クジラ"),
    "cat": ("cat", "cats", "kitten", "kitty", "ねこ", "ネコ", "猫"),
    "dog": ("dog", "dogs", "puppy", "いぬ", "イヌ", "犬"),
    "apple": ("apple", "apples", "りんご", "リンゴ"),
    "robot": ("robot", "robots", "android", "ロボット"),
    "rabbit": ("rabbit", "rabbits", "bunny", "うさぎ", "兎"),
    "horse": ("horse", "horses", "pony", "馬"),
    "bear": ("bear", "bears", "熊"),
    "fox": ("fox", "foxes", "きつね", "狐"),
    "owl": ("owl", "owls", "ふくろう"),
    "butterfly": ("butterfly", "butterflies", "蝶"),
    "star": ("star", "stars", "星"),
    "sun": ("sun", "sunny", "太陽"),
    "moon": ("moon", "lunar", "月"),
    "car": ("car", "cars", "auto", "automobile", "vehicle", "車"),
    "bus": ("bus", "buses", "バス"),
    "bicycle": ("bicycle", "bicycles", "bike", "cycle", "自転車"),
    "airplane": ("airplane", "airplanes", "plane", "aircraft", "飛行機"),
    "boat": ("boat", "boats", "ship", "船"),
    "tree": ("tree", "trees", "forest", "木", "森"),
    "mountain": ("mountain", "mountains", "山"),
    "cloud": ("cloud", "clouds", "雲"),
    "flower": ("flower", "flowers", "floral", "blossom", "rose", "tulip", "sunflower", "daisy", "orchid", "花"),
    "house": ("house", "houses", "home", "building", "家"),
    "bird": ("bird", "birds", "cardinal", "peacock", "parrot", "eagle", "sparrow", "鳥"),
    "fish": ("fish", "fishes", "魚"),
    "train": ("train", "trains", "railway", "電車", "列車"),
    "castle": ("castle", "castles", "城"),
    "banana": ("banana", "bananas", "バナナ"),
    "orange": ("orange", "oranges", "オレンジ"),
    "strawberry": ("strawberry", "strawberries", "いちご"),
    "cake": ("cake", "cakes", "ケーキ"),
    "pizza": ("pizza", "pizzas", "ピザ"),
    "bread": ("bread", "loaf", "パン"),
    "book": ("book", "books", "本"),
    "chair": ("chair", "chairs", "椅子"),
    "clock": ("clock", "clocks", "watch", "時計"),
    "cup": ("cup", "cups", "mug", "コップ"),
    "mushroom": ("mushroom", "mushrooms", "きのこ"),
    "heart": ("heart", "hearts", "ハート"),
    "ball": ("ball", "balls", "ボール"),
    "guitar": ("guitar", "guitars", "ギター"),
    "camera": ("camera", "cameras", "カメラ"),
    "shoe": ("shoe", "shoes", "sneaker", "靴"),
    "umbrella": ("umbrella", "umbrellas", "傘", "かさ"),
    "key": ("key", "keys", "鍵", "かぎ"),
    "bottle": ("bottle", "bottles", "ボトル", "瓶"),
    "pencil": ("pencil", "pencils", "鉛筆", "えんぴつ"),
    "lamp": ("lamp", "lamps", "light", "ライト", "ランプ"),
    "phone": ("phone", "phones", "smartphone", "スマホ", "携帯"),
    "computer": ("computer", "computers", "laptop", "pc", "パソコン"),
    "crown": ("crown", "crowns", "王冠"),
    "diamond": ("diamond", "diamonds", "gem", "jewel", "宝石", "ダイヤ"),
    "sword": ("sword", "swords", "剣"),
    "shield": ("shield", "shields", "盾"),
    "cactus": ("cactus", "cacti", "サボテン"),
    "volcano": ("volcano", "volcanoes", "火山"),
    "fire": ("fire", "flame", "flames", "炎", "火"),
    "icecream": ("icecream", "ice cream", "ice-cream", "アイス", "アイスクリーム"),
    "donut": ("donut", "donuts", "doughnut", "doughnuts", "ドーナツ"),
    "sushi": ("sushi", "寿司", "すし"),
    "face": ("face", "faces", "portrait", "person", "girl", "boy", "顔", "人物", "女の子", "男の子"),
}

COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "scarlet", "赤", "赤い"),
    "orange": ("orange", "橙", "オレンジ色"),
    "yellow": ("yellow", "gold", "golden", "黄色", "金色"),
    "green": ("green", "緑", "緑色"),
    "blue": ("blue", "cyan", "青", "青い", "水色"),
    "purple": ("purple", "violet", "紫"),
    "pink": ("pink", "ピンク"),
    "brown": ("brown", "茶色"),
    "black": ("black", "黒", "黒い"),
    "white": ("white", "白", "白い"),
    "gray": ("gray", "grey", "silver", "銀色", "灰色"),
}

ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "sitting": ("sitting", "sit", "seated", "座る", "座っている"),
    "standing": ("standing", "stand", "立つ", "立っている"),
    "running": ("running", "run", "走る", "走っている"),
    "walking": ("walking", "walk", "歩く", "歩いている"),
    "flying": ("flying", "fly", "飛ぶ", "飛んでいる"),
    "swimming": ("swimming", "swim", "泳ぐ", "泳いでいる"),
    "sleeping": ("sleeping", "sleep", "眠る", "寝ている"),
    "eating": ("eating", "eat", "食べる", "食べている"),
    "holding": ("holding", "hold", "持つ", "持っている"),
    "wearing": ("wearing", "wear", "着る", "着ている"),
    "jumping": ("jumping", "jump", "跳ぶ", "ジャンプ"),
    "parked": ("parked", "parking", "駐車"),
    "floating": ("floating", "float", "drifting", "drift", "浮く", "浮いている"),
    "tilted": ("tilted", "tilted slightly", "leaning", "lean", "swaying", "sway", "turning", "turn", "傾く", "斜め"),
    "shining": ("shining", "shine", "glowing", "glow", "sparkling", "sparkle", "輝く", "光る"),
    "rolling": ("rolling", "roll", "転がる", "転がっている"),
    "bouncing": ("bouncing", "bounce", "跳ねる", "弾む"),
    "smiling": ("smiling", "smile", "happy", "笑顔", "笑う"),
    "looking": ("looking", "look", "facing", "facing forward", "looking left", "looking right", "見る", "向く"),
    "dancing": ("dancing", "dance", "踊る", "踊っている"),
    "climbing": ("climbing", "climb", "登る", "登っている"),
    "spinning": ("spinning", "spin", "回る", "回転"),
    "sliding": ("sliding", "slide", "滑る", "滑っている"),
    "open": ("open", "opening", "開く", "開いている"),
}

VIEW_ALIASES: dict[str, tuple[str, ...]] = {
    "front": ("front", "front view", "正面"),
    "side": ("side", "side view", "profile", "横向き", "横"),
    "back": ("back", "rear", "back view", "後ろ"),
    "top": ("top", "top view", "overhead", "上から"),
    "closeup": ("closeup", "close up", "close-up", "macro", "アップ"),
}

MODIFIER_ALIASES: dict[str, tuple[str, ...]] = {
    "small": ("small", "tiny", "mini", "little", "小さい", "小さな"),
    "large": ("large", "big", "giant", "huge", "大きい", "大きな"),
    "cute": ("cute", "kawaii", "かわいい"),
    "simple": ("simple", "clean", "minimal", "plain", "シンプル"),
    "detailed": ("detailed", "intricate", "細かい"),
    "round": ("round", "circular", "丸い"),
    "bright": ("bright", "shiny", "glowing", "明るい"),
    "dark": ("dark", "night", "暗い", "夜"),
    "fluffy": ("fluffy", "soft", "fuzzy", "ふわふわ"),
    "striped": ("striped", "stripe", "stripes", "しましま", "縞"),
    "spotted": ("spotted", "spots", "polka dot", "水玉", "斑点"),
    "open": ("open", "opened", "開いた"),
    "closed": ("closed", "shut", "閉じた"),
    "broken": ("broken", "cracked", "壊れた", "割れた"),
    "wooden": ("wooden", "wood", "木製"),
    "metallic": ("metallic", "metal", "silver", "金属", "金属製"),
    "transparent": ("transparent", "clear", "glass", "透明"),
    "sharp": ("sharp", "pointy", "尖った"),
    "spiky": ("spiky", "spike", "spines", "とげ", "トゲ"),
}

STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "photo": ("photo", "photograph", "realistic", "写真"),
    "anime": ("anime", "manga", "アニメ"),
    "cartoon": ("cartoon", "toon", "comic"),
    "icon": ("icon", "symbol", "emoji", "sticker", "アイコン"),
    "toy": ("toy", "plush", "figurine", "おもちゃ"),
    "watercolor": ("watercolor", "painting", "painted", "水彩"),
    "sketch": ("sketch", "drawing", "lineart", "線画"),
}

COMPOSITE_PROMPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("astronaut horse", ("astronaut", "horse")),
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "with",
    "and",
    "or",
    "to",
    "for",
    "by",
    "at",
    "from",
    "one",
    "single",
    "centered",
    "clear",
    "background",
    "image",
    "picture",
    "illustration",
}

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+")


def normalize_prompt_text(prompt: str) -> str:
    text = prompt.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_prompt(prompt: str) -> list[str]:
    return TOKEN_RE.findall(normalize_prompt_text(prompt))


def canonicalize_prompt(prompt: str) -> str:
    text = normalize_prompt_text(prompt)
    if not text:
        return ""
    tokens = set(tokenize_prompt(text))
    for key, required_keys in COMPOSITE_PROMPTS:
        if all(_has_alias(tokens, required_key) for required_key in required_keys):
            return key
    for key, aliases in PROMPT_ALIASES.items():
        if key == text or _alias_matches(tokens, text, key):
            return key
        if any(_alias_matches(tokens, text, alias) for alias in aliases):
            return key
    return " ".join(tokenize_prompt(text)) or text


@dataclass(frozen=True)
class PromptSlots:
    subjects: tuple[str, ...]
    colors: tuple[str, ...]
    actions: tuple[str, ...]
    views: tuple[str, ...]
    modifiers: tuple[str, ...]
    styles: tuple[str, ...]
    unknown_tokens: tuple[str, ...]

    def phrase_tokens(self, include_views: bool) -> list[str]:
        tokens: list[str] = []
        tokens.extend(self.styles)
        tokens.extend(self.colors)
        tokens.extend(self.modifiers)
        tokens.extend(self.subjects)
        tokens.extend(self.actions)
        if include_views:
            for view in self.views:
                tokens.extend(_view_phrase_tokens(view))
        tokens.extend(self.unknown_tokens)
        return tokens


def normalize_prompt_slots(prompt: str, include_views: bool = True) -> PromptSlots:
    tokens = tokenize_prompt(prompt)
    token_set = set(tokens)
    phrase = " ".join(tokens)
    subjects = tuple(_matched_alias_keys(token_set, phrase, PROMPT_ALIASES))
    colors = tuple(_matched_alias_keys(token_set, phrase, COLOR_ALIASES))
    actions = tuple(_matched_alias_keys(token_set, phrase, ACTION_ALIASES))
    views = tuple(_matched_alias_keys(token_set, phrase, VIEW_ALIASES)) if include_views else ()
    modifiers = tuple(_matched_alias_keys(token_set, phrase, MODIFIER_ALIASES))
    styles = tuple(_matched_alias_keys(token_set, phrase, STYLE_ALIASES))
    recognized = _recognized_tokens(phrase, include_views=include_views)
    has_known_slots = bool(subjects or colors or actions or views or modifiers or styles)
    unknown_tokens = () if has_known_slots else tuple(
        token for token in tokens if token not in STOPWORDS and not _token_is_recognized(token, recognized)
    )
    return PromptSlots(
        subjects=subjects,
        colors=colors,
        actions=actions,
        views=views,
        modifiers=modifiers,
        styles=styles,
        unknown_tokens=unknown_tokens,
    )


def fnv1a_64(value: str) -> int:
    state = FNV_OFFSET_64
    for byte in value.encode("utf-8"):
        state ^= byte
        state = (state * FNV_PRIME_64) & MASK_64
    return state


def xorshift64(state: int) -> int:
    state ^= (state << 13) & MASK_64
    state ^= state >> 7
    state ^= (state << 17) & MASK_64
    return state & MASK_64


def make_prompt_latent(
    prompt: str,
    seed: int,
    count: int,
    encoder: str = PROMPT_ENCODER_HASH,
) -> list[float]:
    if encoder == PROMPT_ENCODER_HASH:
        return _make_hash_latent(prompt, seed, count)
    if encoder == PROMPT_ENCODER_COMPOSITIONAL:
        return _make_compositional_latent(prompt, seed, count, version=1)
    if encoder == PROMPT_ENCODER_COMPOSITIONAL_V2:
        return _make_compositional_latent(prompt, seed, count, version=2)
    raise ValueError(f"unknown prompt encoder: {encoder}")


def _has_alias(tokens: set[str], key: str) -> bool:
    phrase = " ".join(sorted(tokens))
    return _alias_matches(tokens, phrase, key) or any(
        _alias_matches(tokens, phrase, alias) for alias in PROMPT_ALIASES.get(key, ())
    )


def _make_hash_latent(prompt: str, seed: int, count: int) -> list[float]:
    state = (seed & MASK_64) ^ fnv1a_64(canonicalize_prompt(prompt))
    if state == 0:
        state = 0x9E3779B97F4A7C15
    out = []
    for _ in range(count):
        state = xorshift64(state)
        out.append((state & 0xFFFF) / 32767.5 - 1.0)
    return out


def _make_compositional_latent(prompt: str, seed: int, count: int, version: int = 1) -> list[float]:
    if count <= 0:
        return []
    tokens = tokenize_prompt(prompt)
    token_set = set(tokens)
    phrase = " ".join(tokens)
    canonical = canonicalize_prompt(prompt)
    slots = normalize_prompt_slots(prompt, include_views=version >= 2)
    phrase_tokens = slots.phrase_tokens(include_views=version >= 2)
    if phrase_tokens:
        phrase = " ".join(phrase_tokens)
        tokens = phrase_tokens
        token_set = set(tokens)
    values = [0.0] * count

    seed_weight = 0.35 if version == 1 else 0.22
    subject_weight = 0.75 if version == 1 else 0.95
    feature_rounds = 2 if version == 1 else 4
    secondary_rounds = 2 if version == 1 else 3

    _add_seed_noise(values, seed, canonical, seed_weight)
    if slots.subjects:
        for subject in slots.subjects:
            _add_feature(values, f"subject:{subject}", subject_weight, rounds=feature_rounds)
    elif canonical:
        _add_feature(values, f"subject:{canonical}", 0.45, rounds=secondary_rounds)

    for color in slots.colors:
        _add_feature(values, f"color:{color}", 0.60, rounds=secondary_rounds)
    for action in slots.actions:
        _add_feature(values, f"action:{action}", 0.65, rounds=secondary_rounds)
    if version >= 2:
        for view in slots.views:
            _add_feature(values, f"view:{view}", 0.60, rounds=secondary_rounds)
    for modifier in slots.modifiers:
        _add_feature(values, f"modifier:{modifier}", 0.42, rounds=secondary_rounds)
    for style in slots.styles:
        _add_feature(values, f"style:{style}", 0.38, rounds=secondary_rounds)

    recognized = _recognized_tokens(phrase, include_views=version >= 2)
    content_tokens = [token for token in tokens if token not in STOPWORDS and not _token_is_recognized(token, recognized)]
    for token in content_tokens[:8]:
        _add_feature(values, f"token:{token}", 0.18, rounds=2)
    for left, right in zip(tokens, tokens[1:]):
        if left not in STOPWORDS and right not in STOPWORDS:
            _add_feature(values, f"bigram:{left}_{right}", 0.12, rounds=2)
    if phrase:
        _add_feature(values, f"full:{phrase}", 0.10, rounds=2)

    return [max(-1.0, min(1.0, value)) for value in values]


def _add_seed_noise(values: list[float], seed: int, canonical: str, weight: float) -> None:
    state = (seed & MASK_64) ^ fnv1a_64(f"seed:{canonical}")
    if state == 0:
        state = 0x9E3779B97F4A7C15
    for index in range(len(values)):
        state = xorshift64(state)
        values[index] += ((state & 0xFFFF) / 32767.5 - 1.0) * weight


def _add_feature(values: list[float], name: str, weight: float, rounds: int = 2) -> None:
    if not values:
        return
    state = fnv1a_64(name)
    for _ in range(rounds):
        state = xorshift64(state)
        index = state % len(values)
        state = xorshift64(state)
        sign = 1.0 if (state & 1) else -1.0
        values[index] += sign * weight


def _matched_alias_keys(tokens: set[str], phrase: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    matches = []
    for key, words in aliases.items():
        if _alias_matches(tokens, phrase, key) or any(_alias_matches(tokens, phrase, word) for word in words):
            matches.append(key)
    return matches


def _recognized_tokens(phrase: str, include_views: bool = False) -> set[str]:
    recognized: set[str] = set()
    alias_sets = [PROMPT_ALIASES, COLOR_ALIASES, ACTION_ALIASES, MODIFIER_ALIASES, STYLE_ALIASES]
    if include_views:
        alias_sets.append(VIEW_ALIASES)
    for aliases in alias_sets:
        for key, words in aliases.items():
            for value in (key, *words):
                if _alias_matches(set(tokenize_prompt(phrase)), phrase, value):
                    recognized.update(tokenize_prompt(value))
    return recognized


def _token_is_recognized(token: str, recognized: set[str]) -> bool:
    if token in recognized:
        return True
    if token.isascii():
        return False
    return any(part and not part.isascii() and part in token for part in recognized)


def _view_phrase_tokens(view: str) -> list[str]:
    if view in {"front", "side", "back", "top"}:
        return [view, "view"]
    return [view]


def _alias_matches(tokens: set[str], phrase: str, alias: str) -> bool:
    alias_tokens = tokenize_prompt(alias)
    if not alias_tokens:
        return False
    if all(token in tokens for token in alias_tokens):
        return True
    if any(not token.isascii() for token in alias_tokens):
        return normalize_prompt_text(alias).replace(" ", "") in phrase.replace(" ", "")
    return False
