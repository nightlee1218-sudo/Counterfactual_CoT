#!/usr/bin/env python3
"""Build a strict, framework-balanced Gold1000 manifest from manifest 3.

The input manifest is read-only.  This builder produces a new manifest with:
- the user's exact 250 x 4 primary quota and 15 literal subtype quotas;
- exactly 20 official scene families and no domain above 25%;
- canonical source/target CoT text and one-atom source -> target switches;
- generic location labels only (no `*_shelf_*` labels);
- regenerated source/target prompts and preservation metadata.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IN_ROOT = Path(__file__).resolve().parent
OUT_ROOT = IN_ROOT.parent / "gold1000_framework_strict_v4"
OUT = OUT_ROOT / "gold1000_framework_strict_v4.jsonl"

LOCATION_MAP = {
    "upper_shelf_left": "upper_left",
    "upper_shelf_center": "top_region",
    "upper_shelf_right": "upper_right",
    "lower_shelf_left": "lower_left",
    "lower_shelf_center": "bottom_region",
    "lower_shelf_right": "lower_right",
}
ALLOWED_LOCATIONS = {
    "left_region", "center_region", "right_region", "top_region", "bottom_region",
    "upper_left", "upper_right", "lower_left", "lower_right",
}
FURNITURE_STORAGE_FAMILIES = {"bookshelf_study", "closet_wardrobe", "warehouse_shelf"}
DOMAINS = {
    "indoor_counter": {"kitchen_counter", "bathroom_counter", "office_desk", "bedroom_nightstand"},
    "furniture_storage": FURNITURE_STORAGE_FAMILIES,
    "commercial_display": {"grocery_display", "flower_shop", "toolshed_garage", "antique_curio_shop"},
    "outdoor_open": {"city_street", "parking_lot", "playground", "beach_scene"},
    "indoor_activity": {"living_room_play", "classroom", "dining_table"},
    "dense_closeup": {"pantry_shelf_closeup", "picnic_blanket"},
}
DOMAIN_OF_FAMILY = {family: domain for domain, families in DOMAINS.items() for family in families}
DOMAIN_CAP = 250
OFFICIAL_FAMILIES = {
    "kitchen_counter", "bathroom_counter", "office_desk", "bedroom_nightstand",
    "bookshelf_study", "closet_wardrobe", "warehouse_shelf",
    "grocery_display", "flower_shop", "toolshed_garage", "antique_curio_shop",
    "city_street", "parking_lot", "playground", "beach_scene",
    "living_room_play", "classroom", "dining_table",
    "pantry_shelf_closeup", "picnic_blanket",
}
CATEGORY_TARGETS = {
    "paper": ("paper card display", "paper art print", "paper map booklet", "paper note organizer", "paper origami display"),
    "cardboard": ("cardboard storage tray", "cardboard organizer box", "cardboard display stand", "cardboard document holder"),
    "wood": ("wooden display tray", "wooden storage stand", "wooden organizer box", "wooden ornament"),
    "painted wood": ("painted wooden display tray", "painted wooden organizer", "painted wooden ornament"),
    "gilded_wood": ("gilded wooden display tray", "gilded wooden organizer", "gilded wooden ornament"),
    "metal": ("metal display tray", "metal storage stand", "metal organizer", "metal ornament"),
    "metal wire": ("metal wire display tray", "metal wire organizer", "metal wire stand"),
    "brass": ("brass display tray", "brass storage stand", "brass ornament"),
    "leather": ("leather storage pouch", "leather organizer", "leather display case"),
    "woven straw": ("woven straw display basket", "woven straw storage tray", "woven straw organizer"),
    "galvanized_metal": ("galvanized metal tray", "galvanized metal organizer", "galvanized metal stand"),
    "painted_metal": ("painted metal ornament", "painted metal stand", "painted metal container"),
    "plastic": ("plastic storage organizer", "plastic display tray", "plastic utility container"),
    "glass": ("glass display bowl", "glass storage jar", "glass decorative vessel"),
    "ceramic": ("ceramic display bowl", "ceramic storage vessel", "ceramic ornament"),
    "porcelain": ("porcelain display bowl", "porcelain decorative vessel", "porcelain serving cup"),
    "fabric": ("fabric storage pouch", "fabric organizer", "fabric display basket"),
    "cotton": ("cotton storage pouch", "cotton display bag", "cotton organizer"),
    "linen": ("linen storage pouch", "linen organizer", "linen display bag"),
    "wicker": ("wicker storage basket", "wicker display tray", "wicker organizer"),
    "woven rattan": ("woven rattan tray", "woven rattan basket", "woven rattan organizer"),
    "clay": ("clay decorative vessel", "clay storage bowl", "clay garden ornament"),
    "rubber": ("rubber play object", "rubber display object", "rubber organizer"),
    "foam": ("foam play object", "foam display block", "foam organizer"),
    "organic": ("potted plant", "cut flower display", "leafy branch display"),
    "fruit": ("fruit display bowl", "fruit arrangement", "fruit storage display"),
    "plush": ("plush display toy", "plush storage toy", "plush ornament"),
}

CATEGORY_FAMILIES = {
    "same_affordance_replace": {"grocery_display", "classroom", "dining_table", "flower_shop", "toolshed_garage", "warehouse_shelf"},
    "shape_divergent_replace": {"city_street", "parking_lot", "playground"},
    # garden_animals is intentionally omitted from the official 20-scene release.
    "affordance_divergent_replace": {"bookshelf_study", "antique_curio_shop"},
}
GENERIC_CATEGORY_TERMS = {
    "display object", "storage stand", "metal ornament", "ceramic ornament",
    "plastic display tray", "metal display tray", "wooden display tray",
    "cardboard organizer box", "storage vessel",
}
CONTAINER_TERMS = {
    "box", "basket", "bowl", "cup", "jar", "tray", "bin", "drawer",
    "case", "holder", "crate", "vase", "pot", "bucket", "container",
    "pouch", "carton", "tin", "sack", "cage", "sleeve", "bag", "pack",
}
COLLECTIVE_CATEGORY_REPLACEMENTS = {
    "greeting card stack": "greeting card", "folded pajama set": "folded pajama outfit",
    "orange stack": "orange", "bread roll set": "bread roll", "canned soup set": "soup can",
    "potted plant row": "potted plant", "milk carton set": "milk carton",
    "yogurt cup row": "yogurt cup", "butter block row": "butter block",
    "tomato row": "tomato", "banana bunch": "banana", "potato set": "potato",
    "cardboard box stack": "cardboard box", "stacked tire set": "tire",
    "safety helmet set": "safety helmet", "grain sack stack": "grain sack",
    "egg set": "egg", "croissant row": "croissant", "glass jar row": "glass jar",
    "tulip bunch": "tulip flower", "peony bunch": "peony flower",
    "sunflower bunch": "sunflower", "magazine stack": "magazine",
    "vinyl record stack": "vinyl record", "rolled sock pair set": "rolled sock pair",
}
MATERIAL_CATEGORY_WORDS = {
    "wood", "wooden", "metal", "steel", "brass", "copper", "iron", "aluminum",
    "plastic", "glass", "ceramic", "porcelain", "paper", "cardboard", "fabric",
    "cotton", "linen", "cloth", "wicker", "rattan", "clay", "rubber", "foam",
    "leather", "stone", "marble", "wool", "canvas", "silk", "velvet", "denim",
    "wire", "straw",
}
LOCATION_PHRASES = {
    "left_region": "at the left side", "center_region": "in the center", "right_region": "at the right side",
    "top_region": "in the upper area", "bottom_region": "in the lower area",
    "upper_left": "in the upper-left area", "upper_right": "in the upper-right area",
    "lower_left": "in the lower-left area", "lower_right": "in the lower-right area",
}
# Frame-space wording makes a relocation measurable for an image editor: naming
# thirds/halves of the canvas is far more actionable than "left side".
LOCATION_FRAME_HINTS = {
    "left_region": "the left third of the frame", "center_region": "the middle third of the frame",
    "right_region": "the right third of the frame",
    "top_region": "the upper half of the frame", "bottom_region": "the lower half of the frame",
    "upper_left": "the upper-left quadrant of the frame", "upper_right": "the upper-right quadrant of the frame",
    "lower_left": "the lower-left quadrant of the frame", "lower_right": "the lower-right quadrant of the frame",
}
SCENE_CONTEXTS: dict[str, tuple[dict[str, Any], ...]] = {
    "kitchen_counter": ({"category": "wooden cutting board", "color": "natural wood", "material": "wood"}, {"category": "folded kitchen towel", "color": "white", "material": "cotton"}),
    "bathroom_counter": ({"category": "soap dispenser", "color": "white", "material": "ceramic"}, {"category": "folded hand towel", "color": "light blue", "material": "cotton"}),
    "office_desk": ({"category": "closed notebook", "color": "navy", "material": "paper"}, {"category": "desk lamp", "color": "black", "material": "metal"}),
    "bedroom_nightstand": ({"category": "bedside lamp", "color": "cream", "material": "ceramic"}, {"category": "closed book", "color": "blue", "material": "paper"}),
    "bookshelf_study": ({"category": "hardcover book", "color": "navy", "material": "paper"}, {"category": "metal bookend", "color": "black", "material": "metal"}),
    "closet_wardrobe": ({"category": "folded sweater", "color": "beige", "material": "wool"}, {"category": "woven storage basket", "color": "tan", "material": "wicker"}),
    "warehouse_shelf": ({"category": "sealed storage box", "color": "brown", "material": "cardboard"}, {"category": "plastic parts bin", "color": "blue", "material": "plastic"}),
    "grocery_display": ({"category": "woven produce basket", "color": "tan", "material": "wicker"}, {"category": "wooden display crate", "color": "natural wood", "material": "wood"}),
    "flower_shop": ({"category": "ceramic flower vase", "color": "white", "material": "ceramic"}, {"category": "wrapping paper roll", "color": "kraft brown", "material": "paper"}),
    "toolshed_garage": ({"category": "metal toolbox", "color": "red", "material": "metal"}, {"category": "pair of work gloves", "color": "brown", "material": "leather"}),
    "antique_curio_shop": ({"category": "wooden display stand", "color": "dark brown", "material": "wood"}, {"category": "velvet display pad", "color": "burgundy", "material": "velvet"}),
    "city_street": ({"category": "street lamp post", "color": "black", "material": "metal"}, {"category": "public bench", "color": "brown", "material": "wood"}),
    "parking_lot": ({"category": "parking bollard", "color": "yellow", "material": "metal"}, {"category": "painted parking curb", "color": "white", "material": "concrete"}),
    "playground": ({"category": "park bench", "color": "green", "material": "wood"}, {"category": "sandbox bucket", "color": "red", "material": "plastic"}),
    "beach_scene": ({"category": "folded beach towel", "color": "blue", "material": "cotton"}, {"category": "beach umbrella", "color": "yellow", "material": "fabric"}),
    "living_room_play": ({"category": "sofa cushion", "color": "beige", "material": "fabric"}, {"category": "toy storage basket", "color": "tan", "material": "wicker"}),
    "classroom": ({"category": "closed notebook", "color": "blue", "material": "paper"}, {"category": "pencil cup", "color": "yellow", "material": "plastic"}),
    "dining_table": ({"category": "folded cloth napkin", "color": "white", "material": "linen"}, {"category": "ceramic side plate", "color": "white", "material": "ceramic"}),
    "pantry_shelf_closeup": ({"category": "sealed glass storage jar", "color": "clear", "material": "glass"}, {"category": "woven pantry basket", "color": "tan", "material": "wicker"}),
    "picnic_blanket": ({"category": "picnic basket", "color": "tan", "material": "wicker"}, {"category": "folded picnic napkin", "color": "red", "material": "cotton"}),
}
EXTRA_SCENE_CONTEXTS: dict[str, tuple[dict[str, Any], ...]] = {
    "kitchen_counter": ({"category": "fruit bowl", "color": "white", "material": "ceramic"},),
    "bathroom_counter": ({"category": "toothbrush cup", "color": "white", "material": "ceramic"},),
    "office_desk": ({"category": "pencil holder", "color": "gray", "material": "metal"},),
    "bedroom_nightstand": ({"category": "photo frame", "color": "brown", "material": "wood"},),
    "bookshelf_study": (
        {"category": "potted succulent", "color": "green", "material": "ceramic"},
        {"category": "framed photograph", "color": "brown", "material": "wood"},
    ),
    "closet_wardrobe": ({"category": "closed shoe box", "color": "gray", "material": "cardboard"},),
    "warehouse_shelf": ({"category": "packing tape roll", "color": "tan", "material": "plastic"},),
    "grocery_display": ({"category": "reusable shopping bag", "color": "green", "material": "fabric"},),
    "flower_shop": ({"category": "satin ribbon spool", "color": "pink", "material": "fabric"},),
    "toolshed_garage": ({"category": "workbench clamp", "color": "gray", "material": "metal"},),
    "antique_curio_shop": ({"category": "brass magnifying glass", "color": "gold", "material": "brass"},),
    "city_street": ({"category": "bicycle rack", "color": "gray", "material": "metal"},),
    "parking_lot": ({"category": "traffic cone", "color": "orange", "material": "rubber"},),
    "playground": ({"category": "rubber playground ball", "color": "blue", "material": "rubber"},),
    "beach_scene": ({"category": "seashell", "color": "cream", "material": "shell"},),
    "living_room_play": ({"category": "toy car", "color": "red", "material": "plastic"},),
    "classroom": ({"category": "eraser", "color": "pink", "material": "rubber"},),
    "dining_table": ({"category": "water glass", "color": "clear", "material": "glass"},),
    "pantry_shelf_closeup": ({"category": "sealed food tin", "color": "silver", "material": "metal"},),
    "picnic_blanket": ({"category": "reusable water bottle", "color": "green", "material": "metal"},),
}
UNNATURAL_CATEGORY_COMBINATIONS = (
    frozenset({"collectible mug", "magazine folder"}),
    frozenset({"metal bookend", "webcam"}),
)
DEPENDENT_OBJECT_CONTEXTS: dict[str, dict[str, Any]] = {
    "open wicker bottle holder": {"category": "wine bottle", "color": "dark green", "material": "glass"},
    "metal bookend": {"category": "hardcover book", "color": "navy", "material": "paper"},
    "bookend": {"category": "hardcover book", "color": "navy", "material": "paper"},
    "webcam": {"category": "computer monitor", "color": "black", "material": "plastic"},
    "magazine folder": {"category": "magazine", "color": "blue", "material": "paper"},
}
CATEGORY_REPLACEMENTS = {
    "same_affordance_replace": {
        # Same affordance, but the replacement must differ in silhouette at a
        # glance; near-identical swaps (notebook->journal, ruler->measuring
        # stick) produced invisible edits and were replaced. Names avoid hard
        # material words so they stay compatible with the source material.
        "backpack": "shoulder bag", "bolt organizer box": "flip-bin parts organizer",
        "cardboard egg carton": "molded cup carrier with handle", "ceramic plate": "rectangular serving platter",
        "ceramic teapot": "french press coffee maker", "chilled plastic crate": "insulated zippered cooler tote",
        "claw hammer": "rubber mallet", "colored pencil": "chunky marker pen",
        "empty wooden pallet": "rolling platform dolly", "flathead screwdriver": "cordless electric screwdriver",
        "folded jacket": "folded puffer vest", "glass vase": "flower pitcher", "hand saw": "bow saw",
        "hardcover textbook": "ring binder", "mesh produce bag": "stackable produce keeper box",
        "metal display bin": "retail display basket", "metal ruler": "retractable tape measure",
        "metal supply tray": "two-tier desk organizer tray", "open plastic crate": "wheeled storage tote",
        "open plastic gear bin": "open parts box", "open plastic small bin": "open compartment box",
        "open plastic storage bin": "open storage tote", "open wire metal basket": "collapsible storage cube",
        "paper folder": "expanding document organizer", "pencil case": "hard-shell pencil box",
        "plastic classroom chair": "classroom stool", "porcelain teacup": "coffee mug",
        "spiral notebook": "writing clipboard", "steel fork": "deep soup spoon", "steel knife": "kitchen cleaver",
        "textbook": "glossy magazine", "white lily": "potted white orchid", "wicker basket": "lidded laundry hamper",
        "wire display basket": "mesh display tray", "wooden ruler": "retractable tape measure",
        "workbook": "top-spiral notepad", "woven bread basket": "bread serving tray",
    },
    "shape_divergent_replace": {
        "advertising billboard": "digital advertising kiosk", "bicycle rack": "vertical bicycle locker",
        "covered bus stop shelter": "freestanding transit canopy", "fire hydrant": "freestanding fire-suppression valve station",
        "newspaper kiosk": "newspaper vending box", "parked SUV": "parked motorcycle",
        "parked bicycle": "parked mobility scooter", "parked delivery van": "cargo motorcycle",
        "parked pickup truck": "cargo tricycle", "parked sedan car": "parked scooter",
        "pedestrian crossing signal pole": "overhead pedestrian signal gantry",
        "plastic slide": "playground climbing dome", "public bench": "single pedestal park seat",
        "public mailbox": "parcel drop locker", "raised planter box": "tall planter pot",
        "seesaw": "playground swing frame", "spinning carousel platform": "playground climbing pyramid",
        "spring rider toy": "balance beam", "street lamp post": "illuminated bollard",
        "street name sign post": "ground-mounted street signboard", "street tree planter": "tall tree pot",
        "traffic safety barrier": "traffic cone", "trash bin": "wheeled dumpster",
    },
    "affordance_divergent_replace": {
        "antique mantel clock": "decorative pen holder", "antique world globe": "round storage box",
        "brass candlestick": "brass pen holder", "brass oil lamp": "ornamental flower vase",
        "cardboard storage box": "display pedestal", "ceramic vase": "pen-and-note caddy",
        "decorative ceramic urn": "table lamp base", "decorative urn": "pen-and-note caddy",
        "desk clock": "pen cup", "desk lamp": "powered digital photo display", "desktop globe": "round keepsake box",
        "digital alarm clock": "powered digital photo frame", "gilded picture frame": "jewelry tray",
        "glazed ceramic bowl": "decorative paperweight", "hand-painted bone china teacup": "small jewelry dish",
        "hardcover book": "folded-paper shelf sculpture", "leather notebook": "small storage pouch",
        "magazine folder": "single open-front cardboard letter tray", "marble small sculpture": "marble paperweight",
        "ornate hand mirror": "serving tray", "paper storybook": "folded paper sculpture",
        "paperback book": "paper display fan", "pillar candle": "decorative paperweight",
        "porcelain decorative plate": "wall clock face",
        # Keep the same tabletop-decor scale so the swap reads as a different
        # function on the same silhouette family, not an unrelated object.
        "porcelain shepherdess figurine": "pillar candle holder",
        "porcelain vase": "porcelain pen holder", "silver candlestick holder": "desk pen holder",
        "small glass vase": "glass paperweight", "small trophy": "metal bookend",
        "small wall clock": "decorative wall plaque", "vinyl record stack": "vinyl shelf liner",
    },
}


def cp(value: Any) -> Any:
    return copy.deepcopy(value)


def ent(cot: dict[str, Any], eid: str) -> dict[str, Any]:
    return next(e for e in cot["entities"] if e["id"] == eid)


def render(cot: dict[str, Any]) -> str:
    atoms: list[str] = []
    for e in cot["entities"]:
        eid = e["id"]
        atoms += [f"{eid}.object={e['category']}", f"{eid}.count={e['count']}", f"{eid}.location={e['location']}"]
        for key, value in e.get("attributes", {}).items():
            atoms.append(f"{eid}.attribute.{key}={value}")
        for index, relation in enumerate(e.get("relations", [])):
            atoms.append(f"{eid}.relation.{index}={relation['predicate']}:{relation['target']}")
    atoms += [f"background.{key}={value}" for key, value in cot["background"].items()]
    return "; ".join(atoms)


def field_atom(eid: str, field: str) -> str:
    if field == "category": return f"{eid}.object"
    if field in {"count", "location"}: return f"{eid}.{field}"
    if field.startswith("attributes."): return f"{eid}.attribute.{field.split('.', 1)[1]}"
    if field == "relations": return f"{eid}.relation.0"
    raise ValueError(field)


def get_value(cot: dict[str, Any], eid: str, field: str) -> Any:
    e = ent(cot, eid)
    if field == "category": return e["category"]
    if field in {"count", "location"}: return e[field]
    if field.startswith("attributes."): return e["attributes"][field.split(".", 1)[1]]
    if field == "relations":
        r = e["relations"][0]
        return f"{r['predicate']}:{r['target']}"
    raise ValueError(field)


def set_value(cot: dict[str, Any], eid: str, field: str, value: Any) -> None:
    e = ent(cot, eid)
    if field == "category": e["category"] = value
    elif field in {"count", "location"}: e[field] = value
    elif field.startswith("attributes."): e.setdefault("attributes", {})[field.split('.', 1)[1]] = value
    elif field == "relations":
        predicate, target = str(value).split(":", 1)
        e["relations"] = [{"predicate": predicate, "target": target}]
    else: raise ValueError(field)


def normalise_cot(cot: dict[str, Any]) -> None:
    if not isinstance(cot.get("background"), dict):
        cot["background"] = {"scene": str(cot.get("background", "simple scene"))}
    for e in cot["entities"]:
        e["location"] = LOCATION_MAP.get(e["location"], e["location"])
        if e["location"] not in ALLOWED_LOCATIONS:
            raise ValueError(f"unsupported location {e['location']}")
        if e["count"] > 5:
            e["count"] = 5
        if e["count"] < 1:
            e["count"] = 1


def category_target(material: str, old: str, salt: str, attempt: int = 0) -> str:
    choices = CATEGORY_TARGETS.get(material, (f"{material} display object",))
    index = (int(hashlib.sha256((salt + str(attempt)).encode()).hexdigest(), 16) % len(choices))
    result = choices[index]
    return choices[(index + 1) % len(choices)] if result == old else result


MATERIAL_ALIASES = {
    "wood": {"wood", "wooden"}, "painted wood": {"wood", "wooden"}, "gilded_wood": {"gilded", "wood", "wooden"},
    "metal": {"metal"}, "galvanized_metal": {"metal", "galvanized"}, "painted_metal": {"metal", "painted"},
    "steel": {"steel"}, "brass": {"brass"}, "leather": {"leather"}, "plastic": {"plastic"},
    "glass": {"glass"}, "ceramic": {"ceramic"}, "porcelain": {"porcelain"},
    "paper": {"paper", "cardboard"}, "fabric": {"fabric", "cotton", "linen", "cloth"},
    "cotton": {"cotton", "fabric"}, "linen": {"linen", "fabric"}, "wicker": {"wicker"},
    "woven rattan": {"woven", "rattan"}, "woven straw": {"woven", "straw"}, "metal wire": {"metal", "wire"},
    "clay": {"clay"}, "rubber": {"rubber"}, "foam": {"foam"},
    "organic": {"plant", "flower", "leaf", "branch"}, "fruit": {"fruit"}, "plush": {"plush"},
}
EXPLICIT_MATERIAL_WORDS = MATERIAL_CATEGORY_WORDS
ATTACHMENT_BOUND_CATEGORY_TERMS = {"wall-mounted", "ceiling-mounted", "built-in"}
MATERIAL_VISUAL_CUES = {
    "cotton": "smooth fine weave and soft rounded folds",
    "linen": "clearly visible irregular slub weave and crisp angular creases",
    "wool": "thick fuzzy knit texture with visible fibers",
    "silk": "smooth lustrous surface with bright soft highlights",
    "denim": "distinct diagonal twill weave and sturdy seams",
    "leather": "natural grain with subtle creases and a soft sheen",
    "canvas": "coarse plain weave with matte texture",
    "plastic": "smooth molded surface with uniform highlights",
    "metal": "hard reflective surface with sharp specular highlights",
    "stainless_steel": "bright brushed metallic surface with narrow linear reflections",
    "glass": "transparent surface with visible reflections and refraction",
    "ceramic": "opaque glazed surface with a smooth glossy finish",
    "wood": "visible directional wood grain",
    "wicker": "interlaced woven strands",
    "cardboard": "matte fibrous paperboard surface with visible folded edges",
    "paper": "thin matte paper surface with crisp flat edges",
}
MATERIAL_TARGET_OVERRIDES = {
    ("folded t shirt", "cotton", "linen"): "wool",
    ("napkin", "cotton", "linen"): "silk",
}
CATEGORY_VISUAL_DEFINITIONS = {
    "magazine folder": "single upright open-top cardboard magazine file holder with one slanted front opening",
    "single open-front cardboard letter tray": "single shallow open-front cardboard letter tray with one clearly visible compartment, not a box and not a multi-tier rack",
}


def category_visual_minimal_compatible(category: str, material: str) -> bool:
    """Reject category targets that require a preserved material or attachment to change."""
    words = set(category.lower().replace("-", " ").split())
    explicit = words & EXPLICIT_MATERIAL_WORDS
    return (
        not any(term in category.lower() for term in ATTACHMENT_BOUND_CATEGORY_TERMS)
        and (not explicit or bool(explicit & MATERIAL_ALIASES.get(material, {material})))
    )


def natural_value(value: Any) -> str:
    return str(value).replace("_", " ")


def material_visual_cue(material: str) -> str:
    return MATERIAL_VISUAL_CUES.get(material, f"clearly distinct {natural_value(material)} surface texture")


def short_entity_phrase(entity: dict[str, Any]) -> str:
    """Compact name used inside edit instructions.

    The full descriptor repeats material cues and states several times, which
    dilutes the imperative and makes relocation edits unreliable.
    """
    attrs = entity.get("attributes", {})
    display_category = CATEGORY_VISUAL_DEFINITIONS.get(str(entity["category"]), str(entity["category"]))
    category_words = set(display_category.lower().replace("-", " ").split())
    parts: list[str] = []
    color = str(attrs.get("color", "")).strip()
    material = str(attrs.get("material", "")).strip()
    material_words = MATERIAL_ALIASES.get(material, {word for word in natural_value(material).lower().split() if word})
    if color and not (set(natural_value(color).lower().split()) & category_words):
        parts.append(natural_value(color))
    if material and not (material_words & category_words):
        parts.append(natural_value(material))
    parts.append(display_category)
    return " ".join(parts)


def entity_phrase(entity: dict[str, Any]) -> str:
    attrs = entity.get("attributes", {})
    category = str(entity["category"])
    display_category = CATEGORY_VISUAL_DEFINITIONS.get(category, category)
    category_words = set(display_category.lower().replace("-", " ").split())
    parts: list[str] = []
    color = str(attrs.get("color", "")).strip()
    material = str(attrs.get("material", "")).strip()
    material_words = MATERIAL_ALIASES.get(material, {word for word in natural_value(material).lower().split() if word})
    color_words = set(natural_value(color).lower().split())
    if color and not (color_words & category_words) and not color_words <= material_words:
        parts.append(natural_value(color))
    if material and not (material_words & category_words) and not (material_words & color_words):
        parts.append(natural_value(material))
    parts.append(display_category)
    details = []
    if material:
        details.append(f"surface appearance {material_visual_cue(material)}")
    for key in ("pattern", "functional_state", "occlusion_state"):
        value = attrs.get(key)
        if value is not None:
            details.append(f"{key.replace('_', ' ')} {natural_value(value)}")
    descriptor = " ".join(parts)
    if details:
        descriptor += " with " + ", ".join(details)
    return descriptor


def spatial_anchor_guidance(cot: dict[str, Any]) -> str:
    anchor = str(cot.get("background", {}).get("layout_anchor", "")).strip()
    if anchor:
        return anchor
    locations = {str(entity["location"]) for entity in cot["entities"]}
    vertical_locations = {"top_region", "bottom_region", "upper_left", "upper_right", "lower_left", "lower_right"}
    if locations & vertical_locations:
        return "Use at least two clearly visible rigid horizontal shelf levels or compartments. Every object rests fully on a visible physical surface; no object floats in mid-air."
    if locations & {"left_region", "center_region", "right_region"}:
        return "Use one wide, clearly visible continuous support surface or ground plane spanning left to right. Every object rests on that surface, so left/center/right placement is physically measurable."
    return "Use visible stable support surfaces; every object rests naturally and never floats."


def source_prompt(cot: dict[str, Any]) -> str:
    entities = {e["id"]: e for e in cot["entities"]}
    values = []
    for entity in cot["entities"]:
        location = LOCATION_PHRASES.get(str(entity["location"]), natural_value(entity["location"]))
        phrase = f"exactly {entity['count']} {entity_phrase(entity)} {location}"
        for relation in entity.get("relations", []):
            target = entities.get(relation["target"])
            if target is None:
                continue
            if relation["predicate"] == "occlusion_reference":
                state = str(entity.get("attributes", {}).get("occlusion_state", "unoccluded"))
                if state.startswith("partially_occluded_by_"):
                    phrase += (
                        f", partially hidden behind the {entity_phrase(target)}; both objects are visible, "
                        f"the {entity_phrase(target)} is in front, and it covers about one third of this object's outline"
                    )
                else:
                    phrase += (
                        f", fully visible beside the {entity_phrase(target)} with a clear gap and no overlap"
                    )
            else:
                phrase += f", {natural_value(relation['predicate'])} the {entity_phrase(target)}"
        values.append(phrase)
    return (
        f"Create a natural 1024x1024 image of {cot['background'].get('scene', 'a simple scene')}. "
        f"Show {'; '.join(values)}. {spatial_anchor_guidance(cot)} "
        "Every listed object must be distinct, countable, and inside the frame. "
        "Except where a specified occlusion relation requires partial hiding, keep each object clearly recognizable. "
        "Render a native square, full-bleed scene filling all four canvas edges. Never place the scene inside a photo, frame, card, poster, page, border, blank margin, white wedge, collage, or mockup. "
        "Use a level camera and a stable, uncluttered composition. No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
    )


def target_prompt(record: dict[str, Any]) -> str:
    iv = record["intervention"]
    entity = ent(record["source"]["cot"], iv["entity_id"])
    descriptor = entity_phrase(entity)
    old_value, new_value = natural_value(iv["old_value"]), natural_value(iv["new_value"])
    if iv["field"] == "location":
        old_value = LOCATION_PHRASES.get(str(iv["old_value"]), old_value)
        new_value = LOCATION_PHRASES.get(str(iv["new_value"]), new_value)
    if record["subtype"] == "containment":
        old_predicate, container_id = str(iv["old_value"]).split(":", 1)
        new_predicate, _ = str(iv["new_value"]).split(":", 1)
        container = entity_phrase(ent(record["source"]["cot"], container_id))
        old_relation = "inside" if old_predicate == "inside" else "outside and separate from"
        new_relation = "inside" if new_predicate == "inside" else "outside and separate from"
        if new_predicate == "inside":
            outcome = (
                f"After the edit both objects are still present: the whole {descriptor} sits visibly within the {container} opening, "
                f"the {container} rim stays visible, and the {container} keeps its exact original position."
            )
        else:
            outcome = (
                f"After the edit both objects are still present and standing on the same support surface: "
                f"the {descriptor} is fully outside the {container} with a clear empty gap between them, "
                f"and the {container} stays in its exact original position, now visibly empty. Do not delete, shrink, or restyle the {container}."
            )
        action = (
            f"Change only the spatial relation of the {descriptor} from {old_relation} the {container} to {new_relation} the {container}. "
            f"{outcome}"
        )
        preserve = "Preserve all object identities, counts, attributes, background, lighting, camera, and unrelated geometry exactly."
    elif record["subtype"] == "occlusion_state":
        reference = next(relation for relation in entity.get("relations", []) if relation["predicate"] == "occlusion_reference")
        occluder_entity = ent(record["source"]["cot"], reference["target"])
        if not occlusion_is_judgeable(str(entity["category"]), str(occluder_entity["category"])):
            # A small occluder in front of a large object cannot produce a
            # readable overlap; drop this pair instead of shipping a no-op edit.
            record["_semantic_reject"] = (
                f"occluder too small: {occluder_entity['category']} in front of {entity['category']}"
            )
            return record
        occluder = entity_phrase(occluder_entity)
        short = short_entity_phrase(entity)
        short_occluder = short_entity_phrase(ent(record["source"]["cot"], reference["target"]))
        if str(iv["new_value"]) == "unoccluded":
            geometry = (
                f"TASK: separate the two objects so nothing overlaps. "
                f"In the source image the {short} is partly hidden behind the {short_occluder}, so only part of it is visible. "
                f"Step 1: erase the {short_occluder} from its current spot and fill that area with the uninterrupted background. "
                f"Step 2: redraw the same {short_occluder} in a clearly different spot on the same surface, at least one object width away "
                f"from the {short}, with no overlap. "
                f"After the edit both objects remain visible and fully separated, and the {short} is shown completely. "
                "The relative arrangement of the two objects must differ from the source image. Do not return the source image unchanged."
            )
        else:
            geometry = (
                f"TASK: create a visible overlap. "
                f"In the source image the {short} and the {short_occluder} are clearly separated. "
                f"Step 1: erase the {short_occluder} from its current spot and fill that area with the uninterrupted background. "
                f"Step 2: redraw the same {short_occluder} in front of the {short} so it hides roughly one third of the {short}'s outline. "
                f"After the edit both objects remain visible. "
                "The relative arrangement of the two objects must differ from the source image. Do not return the source image unchanged."
            )
        action = f"Change only the visibility of the {descriptor}. {geometry}"
        preserve = "Preserve every object identity, count, non-occlusion attribute, background, lighting, camera, and unrelated geometry exactly."
    elif iv["field"] == "location":
        vertical = any(token in str(iv["old_value"]) + str(iv["new_value"]) for token in ("top", "bottom", "upper", "lower"))
        short = short_entity_phrase(entity)
        old_frame = LOCATION_FRAME_HINTS.get(str(iv["old_value"]), old_value)
        new_frame = LOCATION_FRAME_HINTS.get(str(iv["new_value"]), new_value)
        if vertical:
            geometry = (
                f"TASK: relocate the {short} to a different level. "
                f"In the source image it sits in {old_frame}; in the output it must sit in {new_frame}. "
                f"Step 1: erase the {short} from its current level and let the empty level continue naturally. "
                f"Step 2: redraw the same {short} resting fully on the other visible level, same size and orientation, not floating or tilted. "
                "Its previous level stays visibly empty. Do not return the source image unchanged."
            )
        else:
            geometry = (
                f"TASK: relocate the {short} horizontally across the frame. "
                f"In the source image it sits in {old_frame}; in the output it must sit in {new_frame}, "
                "far enough that the two positions do not overlap at all. "
                f"Step 1: erase the {short} from {old_frame} and let the bare table surface continue naturally there. "
                f"Step 2: redraw the same {short} in {new_frame}, same size and orientation, resting on the same support surface. "
                "Do not return the source image unchanged."
            )
        action = f"Move only the {descriptor} from {old_value} to {new_value}. {geometry}"
        preserve = "Preserve all identities, counts, attributes, background, lighting, camera, and unrelated object positions exactly."
    elif iv["field"] == "category":
        target_definition = CATEGORY_VISUAL_DEFINITIONS.get(str(iv["new_value"]), str(iv["new_value"]))
        entity_count = int(entity.get("count", 1) or 1)
        count_clause = (
            "Replace every instance of it, keeping exactly the same number of instances, positions, color, material, and size."
            if entity_count > 1
            else "Keep it in the exact same position, size, and orientation."
        )
        action = (
            f"Replace only the {descriptor} with a {target_definition}. {count_clause} "
            "Keep its color and material unchanged, and change only its category and the silhouette that category implies. "
            "The new category must be identifiable from its silhouette at a glance; do not turn it into an abstract block, a stack, or a multi-tier object unless that exact category requires it. "
            "Do not move, restyle, or re-light any other object, and keep the background pixel-consistent with the source image."
        )
        preserve = "Preserve every other object identity, count, position, attribute, relation, background, lighting, camera, and composition exactly."
    elif iv["field"] == "attributes.material":
        source_cue = material_visual_cue(str(iv["old_value"]))
        target_cue = material_visual_cue(str(iv["new_value"]))
        action = (
            f"Change only the surface material of the {descriptor} from {old_value} to {new_value}. "
            f"Remove the old {source_cue}; render the new material with {target_cue}. "
            "Keep the object silhouette, color, count, folds, placement, and size unchanged."
        )
        preserve = "Preserve every other object identity, count, position, attribute, relation, background, lighting, camera, and composition exactly."
    elif iv["field"] == "count":
        try:
            old_count, new_count = int(old_value), int(new_value)
        except (TypeError, ValueError):
            old_count, new_count = 0, 0
        if new_count < old_count:
            delta = (
                f"Remove exactly {old_count - new_count} instance(s) of it and let the bare surface show through where it stood; "
                f"keep the remaining {new_count} instance(s) untouched in their exact positions."
            )
        else:
            delta = (
                f"Add exactly {new_count - old_count} more instance(s) of it beside the existing one(s) on free surface with a clear visible gap, "
                "matching their category, color, material, and size."
            )
        action = (
            f"Change only the number of the {descriptor} from {old_value} to {new_value}. {delta} "
            f"After the edit exactly {new_value} instance(s) of it are visible, and every other object in the scene is still present in its exact original position."
        )
        preserve = "Do not delete or add any other object category."
    else:
        if iv["field"] == "attributes.pattern":
            action = (
                f"Change only the surface pattern of the {descriptor} from {old_value} to {new_value}. "
                f"Render the {new_value} pattern in high contrast across the object's entire visible surface so it is obvious at a glance, while keeping silhouette, color, count, and placement unchanged."
            )
        else:
            action = f"Change only the {descriptor}'s {str(iv['field']).split('.', 1)[-1].replace('_', ' ')} from {old_value} to {new_value}. Make the new state obviously different at a glance while keeping silhouette, count, and placement unchanged."
        preserve = "Preserve every other object identity, count, position, attribute, relation, background, lighting, camera, and composition exactly."
    if record["subtype"] in SPATIAL_SUBTYPES or record["subtype"] == "containment":
        integrity = (
            "CRITICAL: the moved object must never appear twice. Its original position is empty in the output "
            "and the object exists exactly once, at the new position. No ghost copy, no leftover duplicate. "
        )
    elif iv["field"] == "attributes.occlusion_state":
        integrity = (
            "CRITICAL: both objects must still be present in the output. Moving one object never means deleting it; "
            "the object you erase from its spot must be redrawn somewhere else in the same scene. "
        )
    else:
        integrity = ""
    return (
        f"{action}. {preserve} {integrity}"
        "The edit must be obvious when the two images are compared side by side. "
        "Keep a native square, full-bleed scene filling the canvas edge to edge. Do not introduce an inset image, photo frame, border, blank margin, white wedge, collage, card, poster, page, or mockup. Do not alter anything beyond this required atomic visual change. No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
    )


def rebuild_metadata(record: dict[str, Any], subtype: str, direction: str | None = None) -> None:
    iv = record["intervention"]
    source, target = record["source"]["cot"], record["target"]["cot"]
    iv["old_value"] = get_value(source, iv["entity_id"], iv["field"])
    iv["new_value"] = get_value(target, iv["entity_id"], iv["field"])
    field = iv["field"]
    dimension = (
        "category_replacement" if field == "category" else
        "instance_cardinality_intervention" if field == "count" else
        "spatial_geometric_intervention" if field in {"location", "relations"} else
        "attribute_intervention"
    )
    record["dimension"] = dimension
    record["subtype"] = subtype
    iv["intervention_type"] = dimension
    iv["intervention_subtype"] = subtype
    iv["operation"] = "replace"
    if direction:
        iv["relation_direction"] = direction
    else:
        iv.pop("relation_direction", None)
    changed = field_atom(iv["entity_id"], field)
    preserve = []
    for atom in render(source).split("; "):
        key = atom.split("=", 1)[0]
        if key == changed: continue
        if key.endswith(".object"): preserve.append(key.replace(".object", ".category"))
        elif ".attribute." in key: preserve.append(key.replace(".attribute.", ".attributes."))
        elif ".relation." in key: preserve.append(key.rsplit(".", 1)[0].replace(".relation", ".relations"))
        else: preserve.append(key)
    allowed = iv["entity_id"] + "." + field
    record["preservation"] = {"allowed_to_change": [allowed], "must_preserve": preserve, "must_revalidate": [allowed]}
    record["source"]["cot_text"] = render(source)
    record["target"]["cot_text"] = render(target)
    record["source"]["prompt"] = source_prompt(source)
    record["target"]["prompt"] = target_prompt(record)


def canonical_subtype(record: dict[str, Any]) -> tuple[str, str | None]:
    iv = record["intervention"]
    attribute_subtypes = {
        "attributes.color": "color", "attributes.material": "material", "attributes.pattern": "pattern",
        "attributes.occlusion_state": "occlusion_state", "attributes.functional_state": "functional_state",
    }
    if iv["field"] in attribute_subtypes:
        return attribute_subtypes[iv["field"]], None
    if iv["field"] == "count":
        return f"{get_value(record['source']['cot'], iv['entity_id'], 'count')}_to_{get_value(record['target']['cot'], iv['entity_id'], 'count')}", None
    if iv["field"] == "location":
        old = str(get_value(record["source"]["cot"], iv["entity_id"], "location"))
        new = str(get_value(record["target"]["cot"], iv["entity_id"], "location"))
        vertical = any(x in old or x in new for x in ("upper", "lower", "top", "bottom"))
        return ("absolute_vertical" if vertical else "absolute_horizontal"), None
    if iv["field"] == "relations":
        old = str(get_value(record["source"]["cot"], iv["entity_id"], "relations"))
        new = str(get_value(record["target"]["cot"], iv["entity_id"], "relations"))
        return "containment", f"{old.split(':', 1)[0]}_to_{new.split(':', 1)[0]}"
    return record.get("subtype", iv.get("intervention_subtype", "color")), None


def fix_relation_pair(record: dict[str, Any]) -> bool:
    """Ensure containment is a replacement, not a relation add/remove."""
    iv = record["intervention"]
    if not str(iv["field"]).startswith("relations"):
        return False
    iv["field"] = "relations"
    source, target = record["source"]["cot"], record["target"]["cot"]
    eid = iv["entity_id"]
    source_entity, target_entity = ent(source, eid), ent(target, eid)
    if not source_entity.get("relations"):
        target_id = target_entity["relations"][0]["target"] if target_entity.get("relations") else next(e["id"] for e in source["entities"] if e["id"] != eid)
        source_entity["relations"] = [{"predicate": "separate_from", "target": target_id}]
        # This source differs from its former group and therefore needs a distinct image.
        record["source_scene_id"] = f"{record['source_scene_id']}__relation_source__{record['pair_id']}"
    source_relation = source_entity["relations"][0]
    if not target_entity.get("relations"):
        target_entity["relations"] = [{"predicate": "inside" if source_relation["predicate"] == "separate_from" else "separate_from", "target": source_relation["target"]}]
    elif target_entity["relations"][0] == source_relation:
        target_entity["relations"] = [{"predicate": "inside" if source_relation["predicate"] == "separate_from" else "separate_from", "target": source_relation["target"]}]
    return True


def fix_location_container(record: dict[str, Any]) -> None:
    iv = record["intervention"]
    if iv["field"] != "location": return
    source = record["source"]["cot"]
    moved = iv["entity_id"]
    contains_child = any(any(r["predicate"] == "inside" and r["target"] == moved for r in e.get("relations", [])) for e in source["entities"])
    if not contains_child: return
    candidates = [e for e in source["entities"] if e["id"] != moved and not e.get("relations")]
    replacement = candidates[0]
    old = replacement["location"]
    new = "right_region" if old != "right_region" else "left_region"
    # Undo the original container move before replacing it with a free object
    # move; otherwise target CoT would contain two changed locations.
    set_value(record["target"]["cot"], moved, "location", ent(source, moved)["location"])
    iv["entity_id"] = replacement["id"]
    set_value(record["target"]["cot"], replacement["id"], "location", new)


def neutralize_material_category(category: str) -> str:
    words = category.replace("-", " ").split()
    kept = [word for word in words if word.lower() not in MATERIAL_CATEGORY_WORDS]
    result = " ".join(kept).strip()
    return result or "object"


def functional_states(category: str) -> tuple[str, str]:
    text = category.lower()
    if any(word in text for word in ("lamp", "clock", "drill")):
        return "off", "on"
    if "soap" in text or "towel" in text:
        return "dry", "wet"
    if "stapler" in text:
        return "unloaded", "loaded"
    if any(word in text for word in ("book", "notebook", "box", "toolbox", "drawer", "crate", "can", "case")):
        return "closed", "open"
    if any(word in text for word in ("wrench", "hammer", "screwdriver", "saw")):
        return "clean", "rusty"
    return "empty", "full"


def repair_occlusion(record: dict[str, Any]) -> None:
    iv = record["intervention"]
    if iv["field"] != "attributes.occlusion_state":
        return
    source, target = record["source"]["cot"], record["target"]["cot"]
    edited = ent(source, iv["entity_id"])
    candidates = [e for e in source["entities"] if e["id"] != edited["id"]]
    if not candidates:
        raise RuntimeError(f"{record['pair_id']}: occlusion requires a reference object")
    reference = min(candidates, key=lambda e: (e["location"] != edited["location"], e["id"]))
    natural_reference = next(
        candidate for candidate in context_candidates(record["template_family"], {edited["category"]})
        if combination_is_natural({edited["category"], candidate["category"]})
    )
    relation = {"predicate": "occlusion_reference", "target": reference["id"]}
    for cot in (source, target):
        current = ent(cot, edited["id"])
        occluder = ent(cot, reference["id"])
        occluder.update({
            "category": natural_reference["category"],
            "count": 1,
            "location": current["location"],
            "attributes": {"color": natural_reference["color"], "material": natural_reference["material"]},
            "relations": [],
        })
        current["relations"] = [r for r in current.get("relations", []) if r["predicate"] != "occlusion_reference"] + [cp(relation)]
    hidden = f"partially_occluded_by_{re.sub(r'[^a-z0-9]+', '_', natural_reference['category'].lower()).strip('_')}"
    reverse = sum(ord(c) for c in record["pair_id"]) % 2 == 1
    set_value(source, edited["id"], iv["field"], hidden if reverse else "unoccluded")
    set_value(target, edited["id"], iv["field"], "unoccluded" if reverse else hidden)
    record["source_scene_id"] = f"{record['source_scene_id']}__occlusion__{record['pair_id']}"


def repair_containment_geometry(record: dict[str, Any]) -> None:
    iv = record["intervention"]
    if iv["field"] != "relations":
        return
    source, target = record["source"]["cot"], record["target"]["cot"]
    relation = ent(source, iv["entity_id"])["relations"][0]
    container_id = relation["target"]
    container = ent(source, container_id)
    if not any(term in container["category"].lower() for term in CONTAINER_TERMS):
        raise RuntimeError(f"{record['pair_id']}: non-container target {container['category']}")
    for cot in (source, target):
        edited = ent(cot, iv["entity_id"])
        container = ent(cot, container_id)
        edited["location"] = container["location"]
        # Keep containment represented in one directed relation only. A stale
        # inverse `container separate_from edited` would contradict `edited inside container`.
        container["relations"] = [
            relation for relation in container.get("relations", [])
            if not (relation["target"] == iv["entity_id"] and relation["predicate"] in {"inside", "separate_from"})
        ]
    record["source_scene_id"] = f"{record['source_scene_id']}__containment__{record['pair_id']}"


def apply_family_layout_anchor(record: dict[str, Any]) -> None:
    family = record["template_family"]
    vertical_families = {"bookshelf_study", "closet_wardrobe", "warehouse_shelf", "living_room_play", "bedroom_nightstand"}
    if family in vertical_families:
        anchor = "Use two clearly visible rigid horizontal shelf levels or cabinet compartments, with an empty area on each level. Every object rests fully on a visible level; no object floats."
    elif family in {"city_street", "parking_lot", "playground", "beach_scene"}:
        anchor = (
            "Use one continuous visible ground plane as the support surface, spanning left to right, "
            "with empty space on both sides. Every object rests on that support surface with clear ground contact, "
            "so any left-to-right placement change is physically measurable. No object floats or hovers."
        )
    else:
        anchor = "Use one wide clearly visible support surface spanning left to right, with empty space on both sides. Every object rests on that surface so placement changes are physically measurable."
    for cot in (record["source"]["cot"], record["target"]["cot"]):
        cot.setdefault("background", {})["layout_anchor"] = anchor


def repair_record_semantics(record: dict[str, Any]) -> None:
    iv = record["intervention"]
    source, target = record["source"]["cot"], record["target"]["cot"]
    eid = iv["entity_id"]
    categories_changed = False
    for cot in (source, target):
        for entity in cot["entities"]:
            replacement = COLLECTIVE_CATEGORY_REPLACEMENTS.get(entity["category"])
            if replacement:
                entity["category"] = replacement
                categories_changed = True
    if categories_changed:
        record["source_scene_id"] = f"{record['source_scene_id']}__singular_categories__{record['pair_id']}"
    if iv["field"] == "attributes.material":
        source_entity = ent(source, eid)
        target_entity = ent(target, eid)
        category = neutralize_material_category(source_entity["category"])
        source_entity["category"] = category
        target_entity["category"] = category
        target_material = str(target_entity.get("attributes", {}).get("material", "")).lower()
        override = MATERIAL_TARGET_OVERRIDES.get((category, str(source_entity.get("attributes", {}).get("material", "")).lower(), target_material))
        if override:
            target_entity["attributes"]["material"] = override
            target_material = override
        color = str(source_entity.get("attributes", {}).get("color", "")).lower()
        if color in {"clear", "transparent", "translucent"} and target_material not in {"glass", "clear plastic", "transparent plastic"}:
            record["_semantic_reject"] = f"transparent color incompatible with {target_material}"
        record["source_scene_id"] = f"{record['source_scene_id']}__material_neutral__{record['pair_id']}"
    elif iv["field"] == "attributes.pattern":
        # plain<->speckled style swaps are invisible in rendered images; force a
        # high-contrast pattern so the edit is observable.
        low_contrast = {"plain", "speckled"}
        old_pattern = str(ent(source, eid).get("attributes", {}).get("pattern", "")).lower()
        new_pattern = str(ent(target, eid).get("attributes", {}).get("pattern", "")).lower()
        if old_pattern in low_contrast and new_pattern in low_contrast and old_pattern != new_pattern:
            ent(target, eid).setdefault("attributes", {})["pattern"] = "striped"
            iv["new_value"] = "striped"
    elif iv["field"] == "attributes.functional_state":
        a, b = functional_states(ent(source, eid)["category"])
        if sum(ord(c) for c in record["pair_id"]) % 2:
            a, b = b, a
        set_value(source, eid, iv["field"], a)
        set_value(target, eid, iv["field"], b)
        record["source_scene_id"] = f"{record['source_scene_id']}__functional__{record['pair_id']}"
    repair_occlusion(record)
    repair_containment_geometry(record)
    apply_family_layout_anchor(record)


def scene_is_clear(record: dict[str, Any]) -> bool:
    for side in ("source", "target"):
        entities = record[side]["cot"]["entities"]
        if sum(int(entity["count"]) for entity in entities) > 8:
            return False
        by_location: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            by_location[str(entity["location"])].append(entity)
        if any(len(group) >= 3 or sum(int(entity["count"]) for entity in group) >= 7 for group in by_location.values()):
            return False
    return True


def normalize_record(original: dict[str, Any]) -> dict[str, Any]:
    record = cp(original)
    normalise_cot(record["source"]["cot"])
    normalise_cot(record["target"]["cot"])
    iv = record["intervention"]
    category_subtypes = {"same_affordance_replace", "shape_divergent_replace", "affordance_divergent_replace"}
    # The framework label is authoritative: legacy rows occasionally labeled a
    # material attribute edit as a category replacement. Convert those rows into
    # a material-compatible category replacement before rendering the target.
    if record.get("subtype") in category_subtypes and iv["field"] != "category":
        iv["field"] = "category"
    if str(iv["field"]).startswith("relations"):
        iv["field"] = "relations"
    # CoT is the authority: only retain the intended atom from target on top of source.
    field, eid = iv["field"], iv["entity_id"]
    source = record["source"]["cot"]
    target_original = record["target"]["cot"]
    target = cp(source)
    try:
        intended = get_value(target_original, eid, field)
    except (KeyError, IndexError):
        intended = None
    record["target"]["cot"] = target
    if intended is not None:
        set_value(target, eid, field, intended)
    relation_rebuilt = fix_relation_pair(record)
    iv = record["intervention"]
    # Every category replacement uses an explicit, human-reviewed mapping whose
    # semantic distance matches the declared subtype. Generic placeholders are
    # never synthesized.
    if iv["field"] == "category":
        old = str(get_value(source, eid, "category"))
        subtype = record.get("subtype", iv.get("intervention_subtype"))
        proposed = CATEGORY_REPLACEMENTS.get(str(subtype), {}).get(old)
        if proposed is None:
            record["_semantic_reject"] = f"no curated {subtype} mapping for {old}"
        else:
            other_categories = {
                str(entity["category"])
                for entity in target["entities"]
                if entity["id"] != eid
            }
            if proposed in other_categories:
                # Replacing X with a category that already exists in the scene
                # would create two indistinguishable instances; reject and let
                # quota backfill produce a different scene instead.
                record["_semantic_reject"] = f"replacement {proposed} already present in scene"
            else:
                set_value(target, eid, "category", proposed)
    repair_record_semantics(record)
    fix_location_container(record)
    subtype, direction = canonical_subtype(record)
    rebuild_metadata(record, subtype, direction)
    return record


def make_bookshelf_replacements() -> list[dict[str, Any]]:
    """Replace 14 garden rows with official bookshelf affordance-divergent records."""
    source = {
        "schema_version": "1.0",
        "entities": [
            {"id": "book_1", "category": "paper storybook", "count": 1, "location": "left_region", "attributes": {"color": "red", "material": "paper"}, "relations": []},
            {"id": "stand_1", "category": "metal book stand", "count": 1, "location": "center_region", "attributes": {"color": "black", "material": "metal"}, "relations": []},
            {"id": "plant_1", "category": "potted plant", "count": 1, "location": "right_region", "attributes": {"color": "green", "material": "organic"}, "relations": []},
        ],
        "background": {"scene": "quiet study bookshelf", "shelf_material": "dark wood", "wall": "plain cream wall"},
    }
    targets = [
        ("book_1", "folded paper sculpture"), ("book_1", "paper lampshade"),
        ("book_1", "paper hand fan"), ("book_1", "paper flower bouquet"),
        ("book_1", "paper storage pouch"),
        ("stand_1", "metal candle holder"), ("stand_1", "metal plant pot"),
        ("stand_1", "metal doorstop"), ("stand_1", "metal pen cup"),
        ("stand_1", "metal jewelry tray"),
        ("plant_1", "fresh herb garnish"), ("plant_1", "leafy table decoration"),
        ("plant_1", "cut flower bouquet"), ("plant_1", "natural leaf wreath"),
    ]
    records = []
    for index, (eid, category) in enumerate(targets, 1):
        target = cp(source)
        set_value(target, eid, "category", category)
        record = {
            "pair_id": f"g1000_fw_bookshelf_affordance_{index:02d}",
            "source_scene_id": "fw_bookshelf_affordance_base",
            "template_family": "bookshelf_study",
            "source": {"cot": cp(source)}, "target": {"cot": target},
            "intervention": {"entity_id": eid, "field": "category", "operation": "replace"},
        }
        rebuild_metadata(record, "affordance_divergent_replace")
        records.append(record)
    return records


def make_classroom_same_records(count: int) -> list[dict[str, Any]]:
    source = {
        "schema_version": "1.0",
        "entities": [
            {"id": "chair_1", "category": "plastic classroom chair", "count": 1, "location": "left_region", "attributes": {"color": "blue", "material": "plastic"}, "relations": []},
            {"id": "tray_1", "category": "metal supply tray", "count": 1, "location": "center_region", "attributes": {"color": "gray", "material": "metal"}, "relations": []},
            {"id": "folder_1", "category": "paper folder", "count": 1, "location": "right_region", "attributes": {"color": "yellow", "material": "paper"}, "relations": []},
        ],
        "background": {"scene": "simple classroom corner", "surface": "light wood desk", "lighting": "soft daylight"},
    }
    changes = [("chair_1", "plastic classroom stool"), ("chair_1", "plastic classroom seat"), ("tray_1", "metal sorting caddy"), ("tray_1", "metal desk tray"), ("folder_1", "paper document sleeve")]
    out = []
    for index, (eid, category) in enumerate(changes[:count], 1):
        target = cp(source); set_value(target, eid, "category", category)
        r = {"pair_id": f"g1000_fw_classroom_same_{index:02d}", "source_scene_id": "fw_classroom_same_base", "template_family": "classroom", "source": {"cot": cp(source)}, "target": {"cot": target}, "intervention": {"entity_id": eid, "field": "category", "operation": "replace"}}
        rebuild_metadata(r, "same_affordance_replace")
        out.append(r)
    return out


SOURCE_REUSE_CAP = 3
ENTITY_COUNT_TARGETS = {1: 200, 2: 500, 3: 300}
# 93 shareable triplets survive the confusable-category filter, so the shared
# pair budget is the largest cap-3 multiple that stays feasible (27.9% sharing).
SHARED_PAIR_TARGET = 279


def cap_source_reuse(records: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Cap how many pairs may share one source_scene_id.

    When a source scene is reused, greedily keep the subset that maximizes
    coverage of distinct `dimension`s first, then distinct `subtype`s, so a
    kept source's surviving pairs are the hardest possible "same image,
    different instruction" contrastive set instead of near-duplicate edits.
    Excess pairs are dropped here; fill_quotas() backfills their subtype
    counts with brand-new, uniquely-suffixed source scenes.
    """
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_scene_id"]].append(record)
    kept: list[dict[str, Any]] = []
    for group in by_source.values():
        if len(group) <= cap:
            kept.extend(group)
            continue
        remaining = list(group)
        selected: list[dict[str, Any]] = []
        seen_dims: set[str] = set()
        seen_subtypes: set[str] = set()
        while len(selected) < cap and remaining:
            def score(rec: dict[str, Any]) -> tuple[int, int]:
                return (
                    0 if rec["dimension"] in seen_dims else 1,
                    0 if rec["subtype"] in seen_subtypes else 1,
                )
            remaining.sort(key=score, reverse=True)
            pick = remaining.pop(0)
            selected.append(pick)
            seen_dims.add(pick["dimension"])
            seen_subtypes.add(pick["subtype"])
        kept.extend(selected)
    return kept


QUOTAS = {
    "same_affordance_replace": 80,
    "shape_divergent_replace": 85,
    "affordance_divergent_replace": 85,
    "color": 45,
    "material": 50,
    "pattern": 50,
    "occlusion_state": 55,
    "functional_state": 50,
    "1_to_2": 60,
    "2_to_1": 60,
    "2_to_3": 70,
    "3_to_2": 60,
    "absolute_horizontal": 75,
    "absolute_vertical": 75,
    "containment": 100,
}


def synthetic_variant(base: dict[str, Any], subtype: str, serial: int, attempt: int) -> dict[str, Any]:
    """Create a new valid transition using an existing canonical source scene."""
    record = cp(base)
    record["pair_id"] = f"g1000_fw_backfill_{subtype}_{serial:03d}_{attempt:02d}"
    original_target = cp(record["target"]["cot"])
    record["target"]["cot"] = cp(record["source"]["cot"])
    source, target = record["source"]["cot"], record["target"]["cot"]
    # A backfill is a new source scene, not a duplicate rewording of an existing
    # transition. Encode a natural, image-relevant background variation in both
    # states; it remains unchanged across the pair.
    outdoor_families = {"city_street", "parking_lot", "playground", "beach_scene"}
    outdoor_lights = ("soft daylight", "warm diffuse sunlight", "cool overcast daylight", "gentle morning light", "soft afternoon light", "even outdoor light", "subtle overcast light")
    indoor_lights = ("soft daylight", "warm diffuse light", "cool diffuse light", "gentle morning light", "soft afternoon light", "even ambient light", "warm indoor light")
    lights = outdoor_lights if record["template_family"] in outdoor_families else indoor_lights
    variant = f"under {lights[serial % len(lights)]}"
    for cot in (source, target):
        cot["background"]["scene"] = f"{cot['background'].get('scene', 'simple scene')}, {variant}"
    record["source_scene_id"] = f"{record['source_scene_id']}__backfill_source_{serial:03d}"

    iv = record["intervention"]
    eid = iv["entity_id"]
    if subtype in {"same_affordance_replace", "shape_divergent_replace", "affordance_divergent_replace"}:
        iv["field"] = "category"
        material = str(ent(source, eid).get("attributes", {}).get("material", "plastic"))
        old = str(get_value(source, eid, "category"))
        # Backfills must use the current reviewed mapping (which now enforces a
        # glance-distinct silhouette); the legacy batch target is only a fallback.
        proposed = CATEGORY_REPLACEMENTS.get(str(subtype), {}).get(old) or str(get_value(original_target, eid, "category"))
        if proposed == old or not category_visual_minimal_compatible(proposed, material):
            raise RuntimeError(f"invalid curated category mapping: {old} -> {proposed}")
        set_value(target, eid, "category", proposed)
    elif subtype in {"color", "material", "pattern", "occlusion_state", "functional_state"}:
        field = f"attributes.{subtype}"
        iv["field"] = field
        old = str(get_value(source, eid, field))
        choices = {
            "color": ("red", "blue", "green", "yellow", "orange", "purple", "black", "white"),
            "material": ("paper", "wood", "metal", "plastic", "glass", "ceramic", "fabric"),
            "pattern": ("plain", "striped", "polka_dot", "checkerboard", "floral", "geometric"),
            "occlusion_state": ("unoccluded", "partially_occluded"),
            "functional_state": functional_states(ent(source, eid)["category"]),
        }[subtype]
        new = choices[(choices.index(old) + attempt + 1) % len(choices)] if old in choices else choices[attempt % len(choices)]
        set_value(target, eid, field, new)
    elif subtype in {"1_to_2", "2_to_1", "2_to_3", "3_to_2"}:
        iv["field"] = "count"
        old, new = map(int, subtype.split("_to_"))
        set_value(source, eid, "count", old)
        set_value(target, eid, "count", new)
        record["source_scene_id"] = f"{record['source_scene_id']}__cardinality_source_{subtype}_{serial:03d}"
    elif subtype in {"absolute_horizontal", "absolute_vertical"}:
        iv["field"] = "location"
        old = str(get_value(source, eid, "location"))
        choices = ("left_region", "center_region", "right_region") if subtype == "absolute_horizontal" else ("top_region", "center_region", "bottom_region")
        if old not in choices:
            old = choices[0]
            set_value(source, eid, "location", old)
            record["source_scene_id"] = f"{record['source_scene_id']}__position_source_{subtype}_{serial:03d}"
        new = choices[(choices.index(old) + attempt + 1) % len(choices)]
        set_value(target, eid, "location", new)
    elif subtype == "containment":
        iv["field"] = "relations"
        relation = ent(source, eid).get("relations", [])
        if not relation:
            target_id = next(e["id"] for e in source["entities"] if e["id"] != eid)
            set_value(source, eid, "relations", f"separate_from:{target_id}")
            record["source_scene_id"] = f"{record['source_scene_id']}__containment_source_{serial:03d}"
        old = str(get_value(source, eid, "relations"))
        predicate, target_id = old.split(":", 1)
        new = ("inside" if predicate == "separate_from" else "separate_from") + ":" + target_id
        set_value(target, eid, "relations", new)
    else:
        raise ValueError(subtype)
    repair_record_semantics(record)
    direction = None
    if subtype == "containment":
        direction = f"{get_value(source, eid, 'relations').split(':', 1)[0]}_to_{get_value(target, eid, 'relations').split(':', 1)[0]}"
    rebuild_metadata(record, subtype, direction)
    return record


def fill_quotas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {(r["source"]["cot_text"], r["target"]["cot_text"]) for r in records}
    by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subtype[record["subtype"]].append(record)
    canonical_bases = {subtype: list(items) for subtype, items in by_subtype.items()}
    domain_counts = Counter(DOMAIN_OF_FAMILY[r["template_family"]] for r in records)
    serial = 1
    for subtype, quota in QUOTAS.items():
        cursor = 0
        while len(by_subtype[subtype]) < quota:
            bases = canonical_bases[subtype]
            # Pattern is supported by flower_shop and closet_wardrobe. During
            # domain balancing, deliberately synthesize replacements from the
            # flower-shop side rather than adding more furniture-storage pairs.
            if subtype == "pattern":
                non_furniture = [record for record in bases if record["template_family"] not in FURNITURE_STORAGE_FAMILIES]
                if non_furniture:
                    bases = non_furniture
            # Prefer bases whose domain has not yet hit the 25% cap, so that
            # source-reuse-cap driven backfilling never pushes a domain over
            # the framework's balance constraint.
            under_cap = [record for record in bases if domain_counts[DOMAIN_OF_FAMILY[record["template_family"]]] < DOMAIN_CAP]
            if under_cap:
                bases = under_cap
            if not bases:
                raise RuntimeError(f"no base record for {subtype}")
            candidate = synthetic_variant(bases[cursor % len(bases)], subtype, serial, cursor // len(bases))
            cursor += 1
            key = (candidate["source"]["cot_text"], candidate["target"]["cot_text"])
            if key in seen or not scene_is_clear(candidate):
                if cursor > len(bases) * 100:
                    raise RuntimeError(f"cannot create unique clear backfill for {subtype}")
                continue
            records.append(candidate)
            by_subtype[subtype].append(candidate)
            domain_counts[DOMAIN_OF_FAMILY[candidate["template_family"]]] += 1
            seen.add(key)
            serial += 1
    return records


def visual_role(category: str) -> str | None:
    text = category.lower()
    if "bookend" in text:
        return "bookend"
    for token, role in (
        ("lamp", "lamp"), ("vase", "vase"), ("clock", "clock"),
        ("notebook", "book"), ("storybook", "book"), ("textbook", "book"),
        ("paperback", "book"), ("hardcover", "book"), ("closed book", "book"),
    ):
        if token in text:
            return role
    return None


MODIFIER_TOKENS = {"a", "of", "with", "and", "the"}


def head_noun(category: str) -> str:
    tokens = [token for token in re.findall(r"[a-z]+", str(category).lower()) if token not in MODIFIER_TOKENS]
    return tokens[-1] if tokens else ""


def categories_confusable(left: str, right: str) -> bool:
    """True when two object names are visually confusable.

    Either they nest (magazine / magazine folder) or they share the same head
    noun (wicker basket / woven produce basket), which makes per-object
    counting and local editing ambiguous.
    """
    left_tokens = set(re.findall(r"[a-z]+", str(left).lower()))
    right_tokens = set(re.findall(r"[a-z]+", str(right).lower()))
    if not left_tokens or not right_tokens:
        return False
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True
    left_head, right_head = head_noun(left), head_noun(right)
    return bool(left_head) and left_head == right_head


def has_confusable_categories(cot: dict[str, Any]) -> bool:
    categories = [str(entity["category"]) for entity in cot["entities"]]
    return any(
        categories_confusable(categories[i], categories[j])
        for i in range(len(categories))
        for j in range(i + 1, len(categories))
    )


def combination_is_natural(categories: set[str]) -> bool:
    roles = [role for category in categories if (role := visual_role(category)) is not None]
    if len(roles) != len(set(roles)):
        return False
    if any(forbidden <= categories for forbidden in UNNATURAL_CATEGORY_COMBINATIONS):
        return False
    if "open wicker bottle holder" in categories and "wine bottle" not in categories:
        return False
    if any("bookend" in category for category in categories) and not any("book" in category and "bookend" not in category for category in categories):
        return False
    if "webcam" in categories and "computer monitor" not in categories:
        return False
    return True


def required_entity_ids(group: list[dict[str, Any]]) -> set[str]:
    required = {record["intervention"]["entity_id"] for record in group}
    changed = True
    while changed:
        changed = False
        for record in group:
            for side in ("source", "target"):
                for entity in record[side]["cot"]["entities"]:
                    if entity["id"] not in required:
                        continue
                    for relation in entity.get("relations", []):
                        if relation["target"] not in required:
                            required.add(relation["target"])
                            changed = True
    return required


SPATIAL_SUBTYPES = {"absolute_horizontal", "absolute_vertical"}


def record_is_count_increase(record: dict[str, Any]) -> bool:
    """True when the intervention adds instances, which needs free scene space."""
    iv = record.get("intervention", {})
    if iv.get("field") != "count":
        return False
    try:
        return int(iv["new_value"]) > int(iv["old_value"])
    except (TypeError, ValueError):
        return False


# Rough scale classes, derived from the object names themselves. They are only
# used to reject edits whose visual footprint would be too small to judge.
LARGE_SCALE_TERMS = {
    "towel", "basket", "crate", "bin", "box", "tray", "bottle", "jar", "pot", "plant",
    "lamp", "chair", "backpack", "bag", "rack", "shelf", "stand", "bucket", "vase",
    "urn", "carton", "sack", "bunch", "pile", "stack", "board", "poster", "sign",
}
SMALL_SCALE_TERMS = {
    "spoon", "fork", "knife", "teaspoon", "chalk", "eraser", "pencil", "pen", "crayon",
    "key", "coin", "straw", "toothbrush", "razor", "screw", "bolt", "clip", "ring",
    "bead", "berry", "pebble", "seed", "candle", "swab", "pin", "cap", "cup", "dispenser",
    "mug", "soap", "tube", "stick", "brush", "marker", "eraser",
}


def scale_class(category: str) -> str:
    tokens = set(re.findall(r"[a-z]+", str(category).lower()))
    if tokens & SMALL_SCALE_TERMS:
        return "small"
    if tokens & LARGE_SCALE_TERMS:
        return "large"
    return "medium"


# Minimum share of changed pixels for an edit to count as "visible". A spoon
# moving legitimately repaints far less of the frame than a basket does, so a
# single global threshold either misses small edits or accepts no-op ones.
MIN_CHANGED_RATIO_BY_SCALE = {"small": 0.0015, "medium": 0.004, "large": 0.01}


def annotate_quality_hint(record: dict[str, Any]) -> None:
    """Record the expected edit footprint so image QA can use a fair threshold."""
    entity = ent(record["source"]["cot"], record["intervention"]["entity_id"])
    ratio = MIN_CHANGED_RATIO_BY_SCALE[scale_class(str(entity["category"]))]
    if record["subtype"] == "occlusion_state":
        # Only the overlap band changes, so even large objects score low.
        ratio = min(ratio, 0.0015)
    record["quality_hint"] = {"min_changed_ratio": ratio}


def occlusion_is_judgeable(occluded: str, occluder: str) -> bool:
    """A tiny occluder in front of a large object is invisible in an image."""
    return not (scale_class(occluded) == "large" and scale_class(occluder) == "small")


def normalize_category_replacement_counts(group: list[dict[str, Any]]) -> None:
    """Category swaps read best one-to-one; drop multi-instance replacements.

    "three green notebooks -> three clipboards" is hard to verify and easy for
    an editor to bungle, so the replaced entity is normalised to a single
    instance whenever no count intervention touches the same entity.
    """
    entity_ids = {
        str(record["intervention"]["entity_id"])
        for record in group
        if str(record["intervention"].get("field")) == "category"
    }
    if not entity_ids:
        return
    for eid in entity_ids:
        if any(
            str(record["intervention"].get("field")) == "count"
            and str(record["intervention"].get("entity_id")) == eid
            for record in group
        ):
            continue
        for record in group:
            for side in ("source", "target"):
                entity = next((e for e in record[side]["cot"]["entities"] if e["id"] == eid), None)
                if entity is not None:
                    entity["count"] = 1


def minimum_natural_entity_count(group: list[dict[str, Any]]) -> int:
    categories = {
        ent(record[side]["cot"], record["intervention"]["entity_id"])["category"]
        for record in group for side in ("source", "target")
    }
    if any("bookend" in category for category in categories) or "webcam" in categories or "open wicker bottle holder" in categories:
        return 2
    # A spatial move is only measurable against something that stays put: a
    # single object sliding between two empty levels gives no readable
    # before/after difference, so keep at least one static reference object.
    if any(record["subtype"] in SPATIAL_SUBTYPES for record in group):
        return 2
    return 1


def feasible_keep_sets(group: list[dict[str, Any]], size: int) -> list[tuple[str, ...]]:
    """Return common entity subsets that preserve every pair's relation closure."""
    common_ids = [entity["id"] for entity in group[0]["source"]["cot"]["entities"]]
    if size > len(common_ids):
        return []
    required = required_entity_ids(group)
    feasible: list[tuple[str, ...]] = []
    source_by_id = {entity["id"]: entity for entity in group[0]["source"]["cot"]["entities"]}
    for combo in itertools.combinations(common_ids, size):
        keep = set(combo)
        if not required <= keep:
            continue
        categories = {source_by_id[eid]["category"] for eid in keep}
        if len(group) > 1 and not combination_is_natural(categories):
            continue
        valid = True
        for record in group:
            for side in ("source", "target"):
                entities = record[side]["cot"]["entities"]
                ids = {entity["id"] for entity in entities}
                if not keep <= ids:
                    valid = False
                    break
                for entity in entities:
                    if entity["id"] not in keep:
                        continue
                    if any(relation["target"] not in keep for relation in entity.get("relations", [])):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            feasible.append(combo)
    return feasible


def choose_keep_set(group: list[dict[str, Any]], size: int) -> tuple[str, ...]:
    choices = feasible_keep_sets(group, size)
    if not choices:
        raise RuntimeError(f"no feasible {size}-entity plan for {[r['pair_id'] for r in group]}")
    source_by_id = {entity["id"]: entity for entity in group[0]["source"]["cot"]["entities"]}

    def score(choice: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        locations = {source_by_id[eid]["location"] for eid in choice}
        total_instances = sum(int(source_by_id[eid]["count"]) for eid in choice)
        return len(locations), -total_instances, choice

    return max(choices, key=score)


def context_candidates(family: str, required_categories: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for category in sorted(required_categories):
        dependent = DEPENDENT_OBJECT_CONTEXTS.get(category)
        if dependent is None and "bookend" in category:
            dependent = DEPENDENT_OBJECT_CONTEXTS["bookend"]
        if dependent:
            candidates.append(dependent)
    candidates.extend(SCENE_CONTEXTS[family])
    candidates.extend(EXTRA_SCENE_CONTEXTS[family])
    return candidates


def curate_group_entities(group: list[dict[str, Any]], size: int) -> None:
    """Keep only semantically required entities and fill remaining slots with natural scene props."""
    required = required_entity_ids(group)
    if len(required) > size:
        raise RuntimeError(f"required entities exceed size={size}: {[r['pair_id'] for r in group]}")
    source_entities = {entity["id"]: entity for entity in group[0]["source"]["cot"]["entities"]}
    required_categories = {
        ent(record[side]["cot"], eid)["category"]
        for record in group for side in ("source", "target") for eid in required
    }
    missing_slots = size - len(required)
    used_categories = set(required_categories)
    used_roles = {role for category in required_categories if (role := visual_role(category)) is not None}
    original_contexts = [
        {
            "category": entity["category"],
            "color": str(entity.get("attributes", {}).get("color", "neutral")),
            "material": str(entity.get("attributes", {}).get("material", "unspecified")),
        }
        for entity in group[0]["source"]["cot"]["entities"]
        if entity["id"] not in required
    ]
    available_contexts = context_candidates(group[0]["template_family"], required_categories) + original_contexts
    contexts: list[dict[str, Any]] = []
    for candidate in available_contexts if missing_slots else ():
        if candidate["category"] in used_categories:
            continue
        if any(categories_confusable(candidate["category"], category) for category in used_categories):
            continue
        role = visual_role(candidate["category"])
        if role is not None and role in used_roles:
            continue
        proposed = used_categories | {candidate["category"]}
        if not combination_is_natural(proposed):
            continue
        selected_context_categories = {context["category"] for context in contexts}
        if any(
            not combination_is_natural(
                {ent(record[side]["cot"], eid)["category"] for eid in required}
                | selected_context_categories | {candidate["category"]}
            )
            for record in group for side in ("source", "target")
        ):
            continue
        contexts.append(candidate)
        used_categories.add(candidate["category"])
        if role is not None:
            used_roles.add(role)
        if len(contexts) == missing_slots:
            break
    if len(contexts) != missing_slots:
        raise RuntimeError(
            f"insufficient natural contexts for {[r['pair_id'] for r in group]}; "
            f"size={size}, required={sorted(required_categories)}, selected={contexts}, available={available_contexts}"
        )

    occupied = {source_entities[eid]["location"] for eid in required}
    location_order = ["left_region", "center_region", "right_region", "upper_left", "upper_right", "bottom_region"]
    context_entities: list[dict[str, Any]] = []
    for index, candidate in enumerate(contexts, start=1):
        location = next((value for value in location_order if value not in occupied), location_order[index % len(location_order)])
        occupied.add(location)
        context_entities.append({
            "id": f"context_{index}",
            "category": candidate["category"],
            "count": 1,
            "location": location,
            "attributes": {"color": candidate["color"], "material": candidate["material"]},
            "relations": [],
        })

    for record in group:
        for side in ("source", "target"):
            cot = record[side]["cot"]
            retained = [entity for entity in cot["entities"] if entity["id"] in required]
            cot["entities"] = retained + cp(context_entities)
    for record in group:
        for side in ("source", "target"):
            final_categories = {entity["category"] for entity in record[side]["cot"]["entities"]}
            if not combination_is_natural(final_categories):
                raise RuntimeError(f"unnatural curated combination: {record['pair_id']}:{side}:{sorted(final_categories)}")


def shared_triplets(records: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], set[str]]:
    """Select exactly 100 semantically diverse, feasible three-pair source groups."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["source_scene_id"]].append(record)
    candidates: list[tuple[tuple[int, int, str], list[dict[str, Any]]]] = []
    for scene_id, group in groups.items():
        if len(group) < 3:
            continue
        best: tuple[tuple[int, int, str], list[dict[str, Any]]] | None = None
        for combo in itertools.combinations(group, 3):
            subset = list(combo)
            if not any(feasible_keep_sets(subset, size) for size in (2, 3)):
                continue
            diversity = (len({r["dimension"] for r in subset}), len({r["subtype"] for r in subset}), scene_id)
            if best is None or diversity > best[0]:
                best = (diversity, subset)
        if best is not None:
            candidates.append(best)
    candidates.sort(key=lambda item: item[0], reverse=True)
    needed_groups = SHARED_PAIR_TARGET // SOURCE_REUSE_CAP
    if SHARED_PAIR_TARGET % SOURCE_REUSE_CAP or len(candidates) < needed_groups:
        raise RuntimeError(f"cannot form {SHARED_PAIR_TARGET} shared pairs as cap-{SOURCE_REUSE_CAP} groups; candidates={len(candidates)}")
    selected = [group for _, group in candidates[:needed_groups]]
    selected_ids = {record["pair_id"] for group in selected for record in group}
    return selected, selected_ids


def add_source_variant(group: list[dict[str, Any]], index: int) -> None:
    views = (
        "viewed straight-on at eye level", "viewed from a gentle left three-quarter angle",
        "viewed from a gentle right three-quarter angle", "viewed from a slightly elevated angle",
        "viewed from a slightly lower angle",
    )
    spacing = ("with balanced spacing", "with generous spacing", "with compact but uncluttered spacing")
    depth = ("with shallow scene depth", "with moderate scene depth", "with a gently layered depth", "with clear foreground separation")
    illumination = ("with soft balanced illumination", "with gentle even illumination", "with crisp natural illumination", "with mild directional illumination", "with diffuse illumination", "with calm neutral illumination", "with clear ambient illumination")
    framing = ("using centered framing", "using slightly left-weighted framing", "using slightly right-weighted framing")
    variant = ", ".join((
        views[index % len(views)],
        spacing[(index // len(views)) % len(spacing)],
        depth[(index // (len(views) * len(spacing))) % len(depth)],
        illumination[(index // (len(views) * len(spacing) * len(depth))) % len(illumination)],
        framing[(index // (len(views) * len(spacing) * len(depth) * len(illumination))) % len(framing)],
    ))
    for record in group:
        for side in ("source", "target"):
            background = record[side]["cot"]["background"]
            background["scene"] = f"{background.get('scene', 'simple scene')}, {variant}"


def redesign_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a 20/50/30 one/two/three-entity curriculum and 30% shared pairs."""
    shared, shared_ids = shared_triplets(records)
    units: list[list[dict[str, Any]]] = shared + [[record] for record in records if record["pair_id"] not in shared_ids]
    expected_shared_groups = SHARED_PAIR_TARGET // SOURCE_REUSE_CAP
    expected_units = 1000 - SHARED_PAIR_TARGET + expected_shared_groups
    if len(shared) != expected_shared_groups or len(units) != expected_units:
        raise RuntimeError(f"unexpected sharing plan: shared_groups={len(shared)}, units={len(units)}")

    feasible: dict[int, set[int]] = {}
    for unit_index, group in enumerate(units):
        normalize_category_replacement_counts(group)
        minimum_size = minimum_natural_entity_count(group)
        feasible[unit_index] = {size for size in range(minimum_size, 4) if feasible_keep_sets(group, size)}
        if any(record_is_count_increase(record) for record in group):
            # Adding an instance needs unoccupied surface to land on; prefer a
            # two-entity scene. Best effort: scenes that structurally require
            # three entities keep their only feasible size.
            capped = {size for size in feasible[unit_index] if size <= 2}
            if capped:
                feasible[unit_index] = capped
            # Otherwise every size below three is structurally impossible for
            # this group; keep the three-entity plan instead of failing the build.
        if not feasible[unit_index]:
            raise RuntimeError(f"no feasible entity plan for {[r['pair_id'] for r in group]}")

    assignments: dict[int, int] = {}
    forced_three = [index for index in range(len(units)) if min(feasible[index]) == 3]
    forced_weight = sum(len(units[index]) for index in forced_three)
    if forced_weight > ENTITY_COUNT_TARGETS[3]:
        raise RuntimeError(f"forced three-entity pairs={forced_weight}")
    assignments.update({index: 3 for index in forced_three})

    three_needed = ENTITY_COUNT_TARGETS[3] - forced_weight
    for index, group in enumerate(units):
        if three_needed == 0:
            break
        if index in assignments or len(group) != 1 or 3 not in feasible[index]:
            continue
        assignments[index] = 3
        three_needed -= 1
    if three_needed:
        raise RuntimeError(f"cannot allocate exact three-entity target, remaining={three_needed}")

    one_needed = ENTITY_COUNT_TARGETS[1]
    for index, group in enumerate(units):
        if one_needed == 0:
            break
        if index in assignments or len(group) != 1 or 1 not in feasible[index]:
            continue
        assignments[index] = 1
        one_needed -= 1
    if one_needed:
        raise RuntimeError(f"cannot allocate exact one-entity target, remaining={one_needed}")

    for index in range(len(units)):
        if index not in assignments:
            if 2 not in feasible[index]:
                raise RuntimeError(f"remaining unit cannot use two entities: {[r['pair_id'] for r in units[index]]}")
            assignments[index] = 2

    pair_histogram = Counter()
    rebuilt: list[dict[str, Any]] = []
    source_signature_to_id: dict[str, str] = {}
    for unit_index, group in enumerate(units):
        size = assignments[unit_index]
        choose_keep_set(group, size)  # fail early if no relation-closed natural plan exists
        curate_group_entities(group, size)
        original_scene_id = group[0]["source_scene_id"]
        add_source_variant(group, unit_index)
        new_scene_id = f"{original_scene_id}__curriculum_{unit_index:04d}"
        for record in group:
            record["source_scene_id"] = new_scene_id
            rebuild_metadata(record, record["subtype"], record["intervention"].get("relation_direction"))
            pair_histogram[size] += 1
            rebuilt.append(record)
        signature = group[0]["source"]["cot_text"] + "\n" + group[0]["source"]["prompt"]
        previous_id = source_signature_to_id.get(signature)
        if previous_id is not None and previous_id != new_scene_id:
            raise RuntimeError(f"duplicate source plan across ids: {previous_id}, {new_scene_id}")
        source_signature_to_id[signature] = new_scene_id

    if pair_histogram != Counter(ENTITY_COUNT_TARGETS):
        raise RuntimeError(f"entity curriculum mismatch: {pair_histogram}")
    reuse = Counter(record["source_scene_id"] for record in rebuilt)
    shared_pairs = sum(count for count in reuse.values() if count > 1)
    if shared_pairs != SHARED_PAIR_TARGET or max(reuse.values()) > SOURCE_REUSE_CAP:
        raise RuntimeError(f"sharing mismatch: shared_pairs={shared_pairs}, max={max(reuse.values())}")
    return rebuilt


CODE_PREFIX_RE = re.compile(r"^[a-z]{2,4}\d{0,3}$")


def content_stem(scene_id: str, record: dict[str, Any]) -> str:
    """Human-readable scene stem: drop the short source-batch code, keep the content words."""
    base = scene_id.split("__curriculum_")[0].split("__")[0].strip("_")
    tokens = [token for token in base.split("_") if token]
    if len(tokens) > 1 and CODE_PREFIX_RE.match(tokens[0]):
        tokens = tokens[1:]
    stem = "_".join(tokens).strip("_")
    if len(stem) < 3:
        categories = [
            str(entity["category"]).lower().replace(" ", "_")
            for entity in record["source"]["cot"]["entities"]
        ]
        stem = "_".join(dict.fromkeys(categories))[:60].strip("_")
    return stem or record["template_family"]


def rename_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every pair and source scene a content-descriptive, stable name.

    pair_id:          g1000_<family>__<content_stem>__<subtype>_<NNN>
    source_scene_id:  <family>__<content_stem>__v<NNNN>
    """
    scene_names: dict[str, str] = {}
    for index, scene_id in enumerate(sorted({r["source_scene_id"] for r in records})):
        first = next(r for r in records if r["source_scene_id"] == scene_id)
        stem = content_stem(scene_id, first)
        scene_names[scene_id] = f"{first['template_family']}__{stem}__v{index + 1:04d}"

    pair_names: dict[str, str] = {}
    counters: Counter = Counter()
    for record in records:
        scene_id = record["source_scene_id"]
        stem = scene_names[scene_id].rsplit("__v", 1)[0]
        subtype = str(record["subtype"])
        counters[(stem, subtype)] += 1
        pair_names[record["pair_id"]] = f"g1000_{stem}__{subtype}_{counters[(stem, subtype)]:03d}"

    if len(set(pair_names.values())) != len(records) or len(set(scene_names.values())) != len(scene_names):
        raise RuntimeError("renaming produced duplicate pair or scene ids")
    for record in records:
        record["pair_id"] = pair_names[record["pair_id"]]
        record["source_scene_id"] = scene_names[record["source_scene_id"]]
    return records


def main() -> None:
    raw = []
    for path in sorted(IN_ROOT.glob("*.jsonl")):
        raw += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [normalize_record(record) for record in raw if record["template_family"] != "garden_animals"]
    records = [
        record for record in records
        if "_semantic_reject" not in record
        and scene_is_clear(record)
        and not (has_confusable_categories(record["source"]["cot"]) or has_confusable_categories(record["target"]["cot"]))
        and (
            record["subtype"] not in CATEGORY_FAMILIES
            or record["template_family"] in CATEGORY_FAMILIES[record["subtype"]]
        )
    ]
    records += make_bookshelf_replacements()

    # Shift up to five same-affordance rows from toolshed to classroom. The
    # available count can change after semantic rejection, so keep this dynamic.
    removed = 0
    kept = []
    for record in records:
        if record["template_family"] == "toolshed_garage" and record["subtype"] == "same_affordance_replace" and removed < 5:
            removed += 1
            continue
        kept.append(record)
    records = kept + make_classroom_same_records(removed)

    # Deduplicate transition keys only if a repair accidentally converged records.
    unique, seen = [], set()
    for record in records:
        key = (record["source"]["cot_text"], record["target"]["cot_text"])
        if key not in seen:
            unique.append(record); seen.add(key)
    # Remove twelve furniture-domain pattern records before quota fill. Pattern
    # is also supported by flower_shop; refilling there brings this domain from
    # 26.2% to the framework cap of exactly 25.0% without changing any subtype quota.
    balanced, moved_patterns = [], 0
    for record in unique:
        if (record["template_family"] in FURNITURE_STORAGE_FAMILIES
                and record["subtype"] == "pattern" and moved_patterns < 12):
            moved_patterns += 1
            continue
        balanced.append(record)
    if moved_patterns != 12:
        raise RuntimeError(f"expected 12 furniture pattern records, found {moved_patterns}")

    # Cap same-source reuse so no single source image is stretched across too
    # many edits; excess pairs are dropped here and their subtype quota is
    # backfilled below with brand-new source scenes.
    pre_counts = Counter(r["source_scene_id"] for r in balanced)
    pre_cap_sources, pre_cap_pairs, pre_max = len(pre_counts), len(balanced), max(pre_counts.values())
    balanced = cap_source_reuse(balanced, SOURCE_REUSE_CAP)
    post_counts = Counter(r["source_scene_id"] for r in balanced)
    print(f"source reuse cap={SOURCE_REUSE_CAP}: sources {pre_cap_sources} -> {len(post_counts)}, "
          f"pairs {pre_cap_pairs} -> {len(balanced)}, max_reuse {pre_max} -> {max(post_counts.values())}")

    # Enforce literal framework quotas before backfilling unique variants.
    retained, subtype_counts = [], Counter()
    for record in balanced:
        subtype = record["subtype"]
        if subtype not in QUOTAS:
            raise RuntimeError(f"unexpected framework subtype: {subtype}")
        if subtype_counts[subtype] >= QUOTAS[subtype]:
            continue
        retained.append(record)
        subtype_counts[subtype] += 1
    records = fill_quotas(retained)
    records = redesign_sources(records)
    final_counts = Counter(record["subtype"] for record in records)
    if final_counts != Counter(QUOTAS):
        raise RuntimeError(f"framework quota mismatch: {final_counts}")
    if len(records) != 1000:
        raise RuntimeError(f"records after framework balancing={len(records)}, expected 1000")
    if not {r["template_family"] for r in records}.issubset(OFFICIAL_FAMILIES):
        raise RuntimeError("non-official scene family remains")
    if len({r["template_family"] for r in records}) != 20:
        raise RuntimeError("official scene family count is not 20")
    reuse_counts = Counter(r["source_scene_id"] for r in records)
    max_reuse = max(reuse_counts.values())
    if max_reuse > SOURCE_REUSE_CAP:
        raise RuntimeError(f"source reuse cap violated: max={max_reuse} > {SOURCE_REUSE_CAP}")
    domain_final = Counter(DOMAIN_OF_FAMILY[r["template_family"]] for r in records)
    over = {name: count for name, count in domain_final.items() if count > DOMAIN_CAP}
    if over:
        raise RuntimeError(f"domain cap violated: {over}")
    records = rename_ids(records)
    for record in records:
        annotate_quality_hint(record)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(OUT)
    print(f"unique source scenes: {len(reuse_counts)}, max reuse per source: {max_reuse}")
    print(Counter((r["dimension"], r["subtype"]) for r in records))

if __name__ == "__main__":
    main()
