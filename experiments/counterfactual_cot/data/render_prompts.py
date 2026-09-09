from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from .schema import SceneGraph, load_jsonl, write_jsonl


def render_intervention_text(intervention: Dict[str, Any]) -> str:
    return (
        f"{intervention['intervention_type']}: "
        f"{intervention['entity_id']}.{intervention['field']} "
        f"{intervention['old_value']} -> {intervention['new_value']}"
    )


def render_preservation_text(preservation: Dict[str, Any]) -> str:
    preserve = ", ".join(preservation.get("must_preserve", [])) or "none"
    allowed = ", ".join(preservation.get("allowed_to_change", [])) or "none"
    return f"Preserve: {preserve}. Allowed changes: {allowed}."


def scene_to_prompt(scene: SceneGraph) -> str:
    number_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    def pluralize(category: str, count: int | None) -> str:
        if count == 1 or category.endswith("s"):
            return category
        if category.endswith(("ch", "sh", "x")):
            return category + "es"
        if category.endswith("y") and len(category) > 1 and category[-2] not in "aeiou":
            return category[:-1] + "ies"
        return category + "s"

    entity_by_id = {entity.id: entity for entity in scene.entities}
    sentences = []
    for entity in scene.entities:
        count_text = number_words.get(entity.count, str(entity.count)) if entity.count is not None else "a"
        attributes = " ".join(str(value) for _, value in sorted(entity.attributes.items()))
        category = pluralize(entity.category, entity.count)
        subject = " ".join(part for part in (count_text, attributes, category) if part)
        location = f" in the {entity.location} area" if entity.location else ""
        relation_texts = []
        for relation in entity.relations:
            predicate = str(relation.get("predicate", "")).replace("_", " ").strip()
            target = entity_by_id.get(str(relation.get("target", "")))
            if predicate and target is not None:
                relation_texts.append(f"{predicate} the {target.category}")
        relations = f", {' and '.join(relation_texts)}" if relation_texts else ""
        verb = "is" if entity.count in (None, 1) else "are"
        sentences.append(f"{subject.capitalize()} {verb}{location}{relations}.")

    background = scene.background
    background_clauses = []
    if background.get("scene"):
        background_clauses.append(f"The scene takes place in a {background['scene']}")
    if background.get("surface"):
        background_clauses.append(f"on a {background['surface']}")
    if background.get("floor"):
        background_clauses.append(f"with a {background['floor']}")
    if background.get("ground"):
        background_clauses.append(f"on {background['ground']}")
    if background.get("background"):
        background_clauses.append(f"with {background['background']} in the background")
    if background.get("time"):
        background_clauses.append(f"during {background['time']}")
    if background.get("weather"):
        background_clauses.append(f"in {background['weather']} weather")
    if background.get("lighting"):
        background_clauses.append(f"under {background['lighting']}")
    if background_clauses:
        sentences.append(", ".join(background_clauses) + ".")
    return " ".join(sentences)


def standard_prompt(prompt: str, cot_text: str) -> str:
    return (
        "[MODE=STANDARD_COT]\n"
        f"Prompt: {prompt}\n"
        f"Structured CoT: {cot_text}\n"
        "Generate the image that follows the Structured CoT."
    )


def paired_prompt(pair: Dict[str, Any], include_source_image: bool = True) -> str:
    mode = "COUNTERFACTUAL_EDIT" if include_source_image else "COUNTERFACTUAL_GENERATION"
    return (
        f"[MODE={mode}]\n"
        f"Source Prompt: {pair['source']['prompt']}\n"
        f"Source CoT: {pair['source']['cot_text']}\n"
        f"Intervention: {render_intervention_text(pair['intervention'])}\n"
        f"Target CoT: {pair['target']['cot_text']}\n"
        f"{render_preservation_text(pair.get('preservation', {}))}\n"
        "Generate the target image."
    )


def add_rendered_prompts(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for pair in records:
        pair = dict(pair)
        pair.setdefault("training_prompts", {})
        pair["training_prompts"]["source_standard"] = standard_prompt(pair["source"]["prompt"], pair["source"]["cot_text"])
        pair["training_prompts"]["target_standard"] = standard_prompt(pair["target"]["prompt"], pair["target"]["cot_text"])
        pair["training_prompts"]["paired_edit"] = paired_prompt(pair, include_source_image=True)
        pair["training_prompts"]["paired_generation"] = paired_prompt(pair, include_source_image=False)
        yield pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Render canonical prompts for counterfactual CoT pairs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = load_jsonl(args.input)
    rendered = list(add_rendered_prompts(records))
    write_jsonl(args.output, rendered)
    print(f"Rendered prompts for {len(rendered)} pairs to {args.output}")


if __name__ == "__main__":
    main()