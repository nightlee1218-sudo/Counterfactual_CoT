#!/usr/bin/env python3
"""E1 local VQA scorer: T2I-CompBench-style question answering with BLIP-2.

Pure-local replacement for the e1_general_t2i API-based score_vqa.py (venus
llmproxy -> HTTP 403 on the eval instance): the same vqa_{model}.jsonl row
contract, but the answerer is Salesforce/blip2-flan-t5-xl loaded from the
local transformers cache / HF (no API, no network calls beyond a possible
one-time weight pull on the GPU instance — precedent: ViT-g-14, dinov2 and
lpips weights were pulled the same way and are cached since).

Model choice (why BLIP-2):
  * /group has no VQA-capable weights (BAGEL-7B-MoT only); the login node
    cannot reach HF, but the GPU instance can (proven by the E1/E3 runs);
  * transformers 4.52.4 (both pythons) natively ships
    Blip2Processor/Blip2ForConditionalGeneration;
  * T2I-CompBench's VQA-based metric itself is built on the BLIP family,
    so question templates stay paper-conventional.

Question templates (per T2I-CompBench):
  attribute_binding (color/shape/texture): one yes/no question per
    (modifier, object) binding, "Is there a {mod} {obj} in the image?";
  counting: "How many {plural obj} are there in the image?" per counted
    object, answer parsed to an integer and compared exactly;
  spatial: "Is there a {objA} {relation} a {objB} in the image?".
Sensitivity extensions (negative controls), default ON, --no-negatives off:
  attribute: swapped-modifier questions between the first two bindings
    ("Is there a pink lipstick?" for "a red lipstick and a pink blush")
    -> attribute leakage fails the item even when the positives pass;
  spatial: mirrored relation for the directional relations
    (right<->left, top<->bottom) -> a "yes" to both directions fails.
  counting has no negative control (paper convention).

Output rows (append-resumable, exactly like e1_general_t2i/score_vqa.py):
  scores_t2i/vqa_{model}.jsonl
    {item_id, subset, source_category, model, vqa_match, n_questions,
     n_correct, parse_error, answers: [{q, expected, got, ok}],
     vqa_model, mode}
stats_threeway.py merges these automatically once >= 2 models have rows
(vqa_match_f binary rate, paired bootstrap like every other component).

CPU selftest: --mock replaces BLIP-2 with a deterministic oracle answerer
(--mock-acc 1.0 = always correct, 0.0 = always wrong), never touches
torch/transformers/PIL nor the image files; the parse coverage / template /
answer-matching tests live in selftest_e3_eval.py.

Run (GPU instance, open_clip-capable python has transformers too):
  /group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python vqa_scorer.py \
      --models base final991 v6 v7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_LEGACY_ROOT = CFCOT / "outputs/e1_general_t2i/gen"
DEFAULT_E3_ROOT = CFCOT / "outputs/e3_eval/gen_t2i"
DEFAULT_OUT_DIR = CFCOT / "outputs/e3_eval/scores_t2i"
DEFAULT_VQA_MODEL = "Salesforce/blip2-flan-t5-xl"

# ---------------------------------------------------------------------------
# vocabularies + templates (extracted from the real 450 E1 prompts,
# e1_general_t2i/prompts/{attribute_binding,counting,spatial_relations}.jsonl)
# ---------------------------------------------------------------------------
COLOR_WORDS = ["red", "pink", "yellow", "green", "brown", "blue", "gold",
               "black", "white", "tan", "silver"]
SHAPE_WORDS = ["teardrop", "conical", "tall", "short", "round", "rectangular",
               "circular", "triangular", "diamond", "big", "small", "cubic",
               "pentagonal", "pyramidal", "cylindrical", "spherical", "oval",
               "square", "oblong", "crescent"]
TEXTURE_WORDS = ["rubber", "glass", "metallic", "wooden", "fabric", "leather",
                 "plastic", "fluffy"]
CATEGORY_VOCAB = {"color": COLOR_WORDS, "shape": SHAPE_WORDS,
                  "texture": TEXTURE_WORDS}

COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8}

# past-tense narration tails in the counting prompts ("... were on the
# blanket", "... created a warm atmosphere"): counts AFTER such a verb are
# only requirements when introduced by a keep-preposition or listed verb
NARRATIVE_VERBS = {
    "is", "are", "was", "were", "be", "been", "completed", "completes",
    "gathered", "gathers", "provided", "provides", "kept", "keeps", "added",
    "adds", "lounged", "lounges", "created", "creates", "played", "plays",
    "chatted", "chats", "entertained", "entertains", "quacked", "quacks",
    "watched", "watches", "filled", "fills", "used", "uses", "raced",
    "races", "packed", "packs", "sang", "sings", "jumped", "jumps",
    "enjoyed", "enjoys", "hopped", "hops", "sailed", "sails", "left",
    "leaves", "adorned", "adorns",
}
COUNT_KEEP_PREPS = {"with", "near", "alongside"}
COUNT_KEEP_PRECEDING = {"carried", "carries"}

# object-extraction stop words for attribute prompts: function words,
# narrative verbs, and the abstract nouns the complex sentences hang the
# modifier on ("a glass dinner party" -> binding dropped, not asked)
OBJ_STOP = {
    "a", "an", "the", "and", "or", "but", "with", "of", "in", "on", "at",
    "to", "for", "as", "by", "from", "that", "its", "their", "is", "are",
    "was", "were", "be", "been", "sits", "sat", "features", "feature",
    "provide", "provides", "provided", "hold", "holds", "held", "view",
    "views", "glide", "glides", "glided", "along", "play", "plays",
    "played", "protect", "protects", "protected", "rain", "opens", "open",
    "closes", "close", "leans", "leaned", "rest", "rests", "rested",
    "pass", "passes", "passed", "illuminate", "illuminates", "illuminated",
    "complement", "complements", "complemented", "mix", "mixes", "mixed",
    "covered", "cover", "covers", "adorned", "adorn", "adorns", "simmers",
    "simmer", "stirs", "stir", "stirred", "clean", "only", "things",
    "visible", "night", "while", "when", "where", "near", "alongside",
    "dinner", "party", "atmosphere", "warm", "lively", "tune", "recipe",
    "picnic", "charm", "comfort", "setup", "guests", "handy", "tool",
    "added", "adds", "added", "perfect", "serving", "snacks",
}
SPATIAL_RELATIONS = ["on the right of", "on the left of", "on the top of",
                     "on the bottom of", "on side of", "next to", "near"]
SPATIAL_MIRROR = {"on the right of": "on the left of",
                  "on the left of": "on the right of",
                  "on the top of": "on the bottom of",
                  "on the bottom of": "on the top of"}

PLURAL_IRREGULAR = {"man": "men", "woman": "women", "person": "people",
                    "mouse": "mice", "sheep": "sheep", "deer": "deer",
                    "fish": "fish", "goose": "geese", "child": "children",
                    "foot": "feet"}


def words(prompt: str) -> list[str]:
    return [w for w in re.split(r"[^a-z]+", prompt.lower()) if w]


def a_an(noun_phrase: str) -> str:
    return "an" if noun_phrase[:1] in "aeiou" else "a"


def pluralize(noun: str) -> str:
    if noun in PLURAL_IRREGULAR:
        return PLURAL_IRREGULAR[noun]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


# ---------------------------------------------------------------------------
# answer parsing
# ---------------------------------------------------------------------------
YES_TOKENS = {"yes", "y", "yeah", "yep", "true", "correct", "right"}
NO_TOKENS = {"no", "n", "nope", "false", "negative", "wrong", "not"}
PARSE_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12, "couple": 2, "dozen": 12}


def parse_yes_no(text: str) -> bool | None:
    toks = [t for t in re.split(r"[^a-z]+", (text or "").lower()) if t]
    if toks and toks[0] in YES_TOKENS:
        return True
    if toks and toks[0] in NO_TOKENS:
        return False
    for t in toks:  # "There is no ..." style fallbacks
        if t in YES_TOKENS:
            return True
        if t in NO_TOKENS:
            return False
    return None


def parse_count(text: str) -> int | None:
    m = re.search(r"\d+", text or "")
    if m:
        return int(m.group(0))
    for t in (t for t in re.split(r"[^a-z]+", (text or "").lower()) if t):
        if t in PARSE_NUMBERS:
            return PARSE_NUMBERS[t]
    return None


def answer_ok(got: str, expected) -> bool:
    if isinstance(expected, bool):
        return parse_yes_no(got) is expected
    if isinstance(expected, int):
        return parse_count(got) == expected
    raise TypeError(f"bad expected answer: {expected!r}")


# ---------------------------------------------------------------------------
# prompt -> questions (T2I-CompBench templates + negative controls)
# ---------------------------------------------------------------------------
def _q(question: str, expected) -> dict:
    return {"q": question, "expected": expected}


def questions_attribute(prompt: str, source_category: str,
                        negatives: bool) -> list[dict]:
    vocab = set(CATEGORY_VOCAB[source_category])
    toks = words(prompt)
    bindings: list[tuple[str, str]] = []
    i = 0
    while i < len(toks):
        if toks[i] in vocab:
            mod = toks[i]
            obj_toks: list[str] = []
            j = i + 1
            while j < len(toks) and len(obj_toks) < 3 and toks[j] not in OBJ_STOP:
                obj_toks.append(toks[j])
                j += 1
            if obj_toks:
                obj = " ".join(obj_toks)
                if j + 1 < len(toks) and toks[j] == "of" and toks[j + 1] not in OBJ_STOP:
                    obj += " of " + toks[j + 1]  # "slice of cake"
                bindings.append((mod, obj))
            i = j
        else:
            i += 1
    if not bindings:
        raise ValueError(f"no {source_category} binding found")
    out: list[dict] = []

    def attr_question(mod: str, obj: str) -> str:
        if obj.endswith("s") and not obj.endswith("ss"):
            return f"Are there {mod} {obj} in the image?"
        return f"Is there {a_an(mod)} {mod} {obj} in the image?"

    for mod, obj in bindings:
        out.append(_q(attr_question(mod, obj), True))
    if negatives and len(bindings) >= 2:
        (m1, o1), (m2, o2) = bindings[0], bindings[1]
        if m1 != m2 and o1 != o2:  # swapped-modifier leakage checks
            out.append(_q(attr_question(m2, o1), False))
            out.append(_q(attr_question(m1, o2), False))
    return out


def questions_counting(prompt: str, negatives: bool) -> list[dict]:
    toks = words(prompt)
    counts: dict[str, int] = {}
    order: list[str] = []
    narrative = False
    for i, t in enumerate(toks):
        if t in NARRATIVE_VERBS:
            narrative = True
            continue
        if t not in COUNT_WORDS:
            continue
        if narrative:
            prev = toks[i - 1] if i else ""
            if prev not in COUNT_KEEP_PREPS and prev not in COUNT_KEEP_PRECEDING:
                continue  # narration tail ("for a picnic"), not a requirement
        if i + 1 >= len(toks) or toks[i + 1] in NARRATIVE_VERBS:
            continue
        obj = toks[i + 1]
        if obj not in counts:
            order.append(obj)
        counts[obj] = COUNT_WORDS[t]
    if not counts:
        raise ValueError("no counted object found")
    out: list[dict] = []
    for obj in order:
        n = counts[obj]
        noun = pluralize(obj) if n == 1 else obj
        out.append(_q(f"How many {noun} are there in the image?", n))
    return out


def questions_spatial(prompt: str, negatives: bool) -> list[dict]:
    text = " ".join(words(prompt))
    obj_pat = r"(?:a|an) ([a-z]+)"
    for rel in sorted(SPATIAL_RELATIONS, key=len, reverse=True):
        m = re.fullmatch(f"{obj_pat} {re.escape(rel)} {obj_pat}", text)
        if m:
            obj_a, obj_b = m.group(1), m.group(2)
            rel_norm = "on the side of" if rel == "on side of" else rel
            out = [_q(f"Is there {a_an(obj_a)} {obj_a} {rel_norm} "
                      f"{a_an(obj_b)} {obj_b} in the image?", True)]
            if negatives and rel in SPATIAL_MIRROR:
                out.append(_q(f"Is there {a_an(obj_a)} {obj_a} {SPATIAL_MIRROR[rel]} "
                              f"{a_an(obj_b)} {obj_b} in the image?", False))
            return out
    raise ValueError(f"no spatial relation matched: {prompt!r}")


def build_questions(record: dict, negatives: bool) -> list[dict]:
    subset = record["subset"]
    prompt = record["prompt"]
    if subset == "attribute_binding":
        return questions_attribute(prompt, record["source_category"], negatives)
    if subset == "counting":
        return questions_counting(prompt, negatives)
    if subset == "spatial_relations":
        return questions_spatial(prompt, negatives)
    raise ValueError(f"unknown subset: {subset!r}")


# ---------------------------------------------------------------------------
# answerers (lazy heavy imports)
# ---------------------------------------------------------------------------
def build_blip2_ctx(args: argparse.Namespace):
    import torch
    from PIL import Image
    from transformers import (Blip2ForConditionalGeneration,
                              Blip2Processor)

    name = str(args.local_vqa_dir or args.vqa_model)
    print(json.dumps({"event": "vqa_load_start", "model": name}), flush=True)
    processor = Blip2Processor.from_pretrained(name)
    dtype = torch.float16 if (args.device != "cpu" and torch.cuda.is_available()) \
        else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(name, torch_dtype=dtype)
    dev = "cpu" if args.device == "cpu" else ("cuda" if torch.cuda.is_available()
                                              else "cpu")
    model = model.to(dev).eval()
    print(json.dumps({"event": "vqa_load_done", "device": dev, "dtype": str(dtype)}),
          flush=True)

    @torch.no_grad()
    def ask(image_path: str, question: str, expected) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, text=f"Question: {question} Answer:",
                           return_tensors="pt").to(dev, dtype)
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=args.max_new_tokens)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    return {"ask": ask, "vqa_model": name, "mode": "blip2"}


def build_mock_ctx(args: argparse.Namespace):
    """Deterministic oracle answerer: correct unless a stable hash of the
    question falls in the flip band (--mock-acc 0 -> every question wrong)."""
    def ask(image_path: str, question: str, expected):
        h = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
        flip = h % 1000 < round((1.0 - args.mock_acc) * 1000)
        if isinstance(expected, bool):
            return ("no" if expected else "yes") if flip else \
                   ("yes" if expected else "no")
        return (str(expected % 12 + 1)) if flip else str(expected)

    return {"ask": ask, "vqa_model": "mock", "mode": "mock"}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def gen_dir_for(model: str, legacy_root: Path, e3_root: Path) -> Path:
    return (e3_root if model in ("v6", "v7") else legacy_root) / model


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help="default: auto-detect models whose "
                             "generation_manifest.jsonl exists (base/final991/v6/v7)")
    parser.add_argument("--gen-legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT,
                        help="E1 generation root holding base/ and final991/")
    parser.add_argument("--gen-e3-root", type=Path, default=DEFAULT_E3_ROOT,
                        help="e3_eval generation root holding v6/ and v7/")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vqa-model", default=DEFAULT_VQA_MODEL)
    parser.add_argument("--local-vqa-dir", type=Path, default=None,
                        help="local weights dir override (offline GPU instance)")
    parser.add_argument("--no-negatives", action="store_true",
                        help="drop the negative-control questions")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N items per model (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--mock", action="store_true",
                        help="CPU selftest mode: deterministic oracle, no "
                             "torch/transformers, no image reads")
    parser.add_argument("--mock-acc", type=float, default=1.0,
                        help="mock accuracy band (1.0 oracle, 0.0 all wrong)")
    args = parser.parse_args()
    if args.mock and not 0.0 <= args.mock_acc <= 1.0:
        parser.error("--mock-acc must be within [0, 1]")
    if args.models:
        bad = [m for m in args.models if m not in ("base", "final991", "v6", "v7")]
        if bad:
            parser.error(f"unknown models: {bad} (base|final991|v6|v7)")
    return args


def main() -> int:
    args = parse_args()
    if args.models is None:
        args.models = [m for m in ("base", "final991", "v6", "v7")
                       if (gen_dir_for(m, args.gen_legacy_root, args.gen_e3_root)
                           / "generation_manifest.jsonl").exists()]
        if not args.models:
            print(json.dumps({"event": "no_manifests_found",
                              "legacy_root": str(args.gen_legacy_root),
                              "e3_root": str(args.gen_e3_root)}), flush=True)
            return 2
        print(json.dumps({"event": "auto_models", "models": args.models}), flush=True)

    ctx = build_mock_ctx(args) if args.mock else build_blip2_ctx(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        manifest = gen_dir_for(model, args.gen_legacy_root, args.gen_e3_root) \
            / "generation_manifest.jsonl"
        records = load_manifest(manifest)
        if args.limit:
            records = records[:args.limit]
        out_path = args.out_dir / f"vqa_{model}.jsonl"
        done: set[str] = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["item_id"])
        written = 0
        with out_path.open("a", encoding="utf-8") as out_f:
            for rec in records:
                if rec["item_id"] in done:
                    continue
                row = {"item_id": rec["item_id"], "subset": rec["subset"],
                       "source_category": rec["source_category"], "model": model}
                try:
                    questions = build_questions(rec, negatives=not args.no_negatives)
                except ValueError as exc:
                    row.update({"vqa_match": None, "n_questions": 0, "n_correct": 0,
                                "parse_error": str(exc), "answers": [],
                                "vqa_model": ctx["vqa_model"], "mode": ctx["mode"]})
                    out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    continue
                answers = []
                for qd in questions:
                    image_path = rec["output_path"]
                    if not args.mock and not Path(image_path).exists():
                        print(json.dumps({"event": "missing_image",
                                          "item_id": rec["item_id"],
                                          "path": image_path}), flush=True)
                        got, ok = "", False
                    else:
                        got = ctx["ask"](image_path, qd["q"], qd["expected"])
                        ok = answer_ok(got, qd["expected"])
                    answers.append({"q": qd["q"],
                                    "expected": qd["expected"],
                                    "got": got, "ok": bool(ok)})
                n_ok = sum(a["ok"] for a in answers)
                row.update({"vqa_match": n_ok == len(answers), "n_questions": len(answers),
                            "n_correct": n_ok, "parse_error": None,
                            "answers": answers, "vqa_model": ctx["vqa_model"],
                            "mode": ctx["mode"]})
                out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                if written % 25 == 0:
                    out_f.flush()
                    print(json.dumps({"event": "progress", "model": model,
                                      "done": written}), flush=True)
        print(json.dumps({"event": "vqa_model_done", "model": model,
                          "rows": written, "out": str(out_path)}), flush=True)
    print(json.dumps({"event": "all_done"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
