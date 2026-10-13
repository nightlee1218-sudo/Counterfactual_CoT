"""E3 v6 token-region alignment machinery (training-forward path, no monkeypatching).

Implements the auxiliary alignment loss

    L_align = -log sigmoid( s * ( cos(h_tok, mu_in) - cos(h_tok, mu_out) ) )

where, per alignment layer L in the LLM backbone (defaults 13/6/16/14, the E2
probe winners), h_tok are the hidden states of the changed CoT value tokens
inside the packed training prompt, and mu_in / mu_out are the mass-weighted
means of the same layer's hidden states over the target-latent positions that
lie inside / outside the automatic edit mask (the v5 objective's shared 64x64
edit/outside partition, row-major).

Extraction happens through ordinary ``nn.Module`` forward hooks installed on
the (unwrapped references of the) selected ``Qwen2MoTDecoderLayer`` modules.
The hooks reduce the full (S, D) layer output to a few small tensors
immediately (row slice + GEMV), so no large intermediate is retained beyond the
activation-checkpoint boundary tensors that the training graph keeps anyway.
Because v5 wraps every decoder layer in a NON-REENTRANT ``CheckpointWrapper``
(``_apply_activation_checkpointing`` in the v3 runtime), the original forward
still runs with autograd enabled and simply discards saved intermediates; the
escaped hook tensors stay connected to the graph, and backward triggers the
standard region recompute.  The CPU selftest verifies gradient parity against a
non-checkpointed baseline.

Span locating is ported from ``e2_token_probe/probe.py`` (anchored '=' / ':'
occurrences of the intervention value) with one training-specific refinement:
only occurrences inside the "Target CoT:" line of the v5 state prompt are used,
because that is the line the model is instructed to follow ("Generate the
target image by strictly following Target CoT.").  For the source_state branch
the same rule anchors ``old_value`` in the source branch's Target CoT.

This module deliberately depends only on torch + stdlib so the CPU selftest can
import it instantly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "ALIGN_DEFAULT_LAYERS",
    "ALIGN_MIN_REGION_MASS",
    "find_changed_occurrences",
    "span_to_token_range",
    "target_cot_occurrences",
    "state_changed_value",
    "changed_token_spans",
    "AlignTargetCache",
    "derive_prompt_layout",
    "build_region_weight_vectors",
    "build_token_index",
    "_weighted_mean",
    "AlignHookManager",
    "align_loss_from_stash",
    "empty_align_metrics",
    "validate_replay_record",
]

# E2 probe winners (final991 > base with paired bootstrap CI): best layers.
ALIGN_DEFAULT_LAYERS = (13, 6, 16, 14)
# A region needs ~2 tokens of partition mass before its mean is meaningful.
ALIGN_MIN_REGION_MASS = 2.0
_EPS = 1e-8


# ---------------------------------------------------------------------------
# span locating (ported from e2_token_probe/probe.py; provenance: E2 probe)
# ---------------------------------------------------------------------------
def find_changed_occurrences(prompt: str, new_value: Any) -> List[Tuple[int, int]]:
    """Occurrences of the intervention value in the rendered prompt.

    Preference: occurrences right after '=' or ':' (CoT assignments,
    e.g. ``vase_1.object=pen-and-note caddy``); fallback: all word-boundary
    occurrences; last resort: plain substring.
    """
    value = str(new_value).strip()
    if not value:
        return []
    pattern = r"(?<![0-9A-Za-z])" + re.escape(value) + r"(?![0-9A-Za-z])"
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, prompt)]
    if not matches:
        idx = prompt.find(value)
        return [(idx, idx + len(value))] if idx >= 0 else []
    anchored = [(s, e) for (s, e) in matches if s > 0 and prompt[s - 1] in "=:."]
    return anchored or matches


def span_to_token_range(tokenizer: Any, prompt: str, start: int, end: int,
                        cache: Optional[Dict[int, int]] = None) -> Tuple[int, int]:
    """Char span -> [first_tok, last_tok) within ``tokenizer.encode(prompt)``.

    Prefix re-encoding keeps this correct for slow (non-offset) tokenizers.
    """
    cache = cache if cache is not None else {}
    for cut in (start, end):
        if cut not in cache:
            cache[cut] = len(tokenizer.encode(prompt[:cut]))
    return cache[start], cache[end]


def target_cot_occurrences(prompt: str, value: Any) -> Tuple[List[Tuple[int, int]], str]:
    """Anchored occurrences of ``value`` restricted to the "Target CoT:" line.

    Falls back to E2 whole-prompt semantics when the line is absent or the
    value does not occur inside it.  Returns (spans, span_source) with
    span_source in {"target_cot", "full_prompt", "none"}.
    """
    occurrences = find_changed_occurrences(prompt, value)
    marker = "Target CoT:"
    pos = prompt.rfind(marker)
    if pos < 0 or not occurrences:
        return occurrences, ("full_prompt" if occurrences else "none")
    tail = pos + len(marker)
    in_line = [(s, e) for (s, e) in occurrences if s >= tail]
    if in_line:
        return in_line, "target_cot"
    return occurrences, ("full_prompt" if occurrences else "none")


def state_changed_value(record: Mapping[str, Any], state_id: str) -> Optional[str]:
    """The CoT value that differs between the two states, for this branch."""
    intervention = record.get("intervention") or {}
    key = "new_value" if state_id == "target_state" else "old_value"
    value = intervention.get(key)
    return str(value) if value is not None else None


def changed_token_spans(tokenizer: Any, prompt: str, value: Any,
                        cache: Optional[Dict[int, int]] = None
                        ) -> Tuple[List[Tuple[int, int]], str]:
    """Merged, non-overlapping token spans of the anchored value occurrences."""
    occurrences, source = target_cot_occurrences(prompt, value)
    if not occurrences:
        return [], source
    tok_cache = cache if cache is not None else {}
    spans = sorted(span_to_token_range(tokenizer, prompt, s, e, tok_cache) for (s, e) in occurrences)
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if b <= a:
            continue
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, source


class AlignTargetCache:
    """Per-(pair_id, state_id) cache of prompt text ids and changed-token spans.

    Avoids re-encoding the ~2KB state prompt every macro step when the record
    sampler revisits a pair.  Simple FIFO eviction keeps memory bounded.
    """

    def __init__(self, tokenizer: Any, capacity: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.capacity = int(capacity)
        self._entries: "Dict[Tuple[str, str], Dict[str, Any]]" = {}
        self._encode_calls = 0

    def entry(self, pair_id: str, state_id: str, prompt: str, value: Any) -> Dict[str, Any]:
        key = (str(pair_id), str(state_id))
        cached = self._entries.get(key)
        if cached is not None and cached["prompt"] == prompt:
            return cached
        text_ids = self.tokenizer.encode(prompt)
        self._encode_calls += 1
        boundary: Dict[int, int] = {}
        spans, span_source = changed_token_spans(self.tokenizer, prompt, value, boundary)
        entry = {
            "pair_id": pair_id,
            "state_id": state_id,
            "prompt": prompt,
            "value": None if value is None else str(value),
            "text_ids": text_ids,
            "spans": spans,
            "span_source": span_source,
        }
        if len(self._entries) >= self.capacity:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = entry
        return entry


# ---------------------------------------------------------------------------
# packed-layout derivation (v5 state sample: [vae_src][vit_src][text][vae_tgt])
# ---------------------------------------------------------------------------
def derive_prompt_layout(packed_text_ids: Sequence[int],
                         packed_vit_token_indexes: Sequence[int],
                         mse_loss_indexes: Sequence[int],
                         prompt_text_ids: Sequence[int]) -> Dict[str, int]:
    """Derive the prompt segment's position from packed batch tensors.

    v5 packs exactly: [SOI + 4096 src-vae + EOI] [SOI + 4900 src-vit + EOI]
    [bos + prompt + eos] [SOI + 4096 tgt-vae + EOI].  From any two adjacent
    boundaries we recover bos_pos and n_text and *verify* the packed text ids
    against ``prompt_text_ids`` so silent layout drift fails fast.
    """
    if len(packed_vit_token_indexes) == 0:
        raise RuntimeError("align: packed sample has no ViT segment (expected v5 state layout)")
    vit_last = int(max(packed_vit_token_indexes))
    first_target_vae = int(mse_loss_indexes[0])
    n_text = first_target_vae - vit_last - 5
    if n_text <= 0:
        raise RuntimeError(f"align: non-positive prompt length {n_text} (layout drift?)")
    if len(packed_text_ids) < 5 + n_text:
        raise RuntimeError("align: packed text ids shorter than expected prompt segment")
    bos_pos = vit_last + 2
    prompt_slice = packed_text_ids[5:5 + n_text]
    if torch.is_tensor(prompt_slice):
        prompt_slice = prompt_slice.tolist()  # single transfer for device tensors
    packed_prompt_ids = [int(v) for v in prompt_slice]
    if packed_prompt_ids != [int(v) for v in prompt_text_ids]:
        raise RuntimeError("align: packed prompt ids differ from tokenizer.encode(state prompt); layout drift")
    return {"bos_pos": bos_pos, "n_text": n_text}


def build_region_weight_vectors(edit_weights: torch.Tensor,
                                outside_weights: torch.Tensor,
                                sequence_length: int,
                                mse_positions: torch.Tensor,
                                device: torch.device,
                                dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-sequence in/out weight vectors with the region masses at the target latents."""
    w_in = torch.zeros(sequence_length, device=device, dtype=dtype)
    w_out = torch.zeros(sequence_length, device=device, dtype=dtype)
    w_in[mse_positions] = edit_weights.to(device=device, dtype=dtype)
    w_out[mse_positions] = outside_weights.to(device=device, dtype=dtype)
    return w_in, w_out


def build_token_index(spans: Sequence[Tuple[int, int]], bos_pos: int, device: torch.device) -> torch.Tensor:
    """Absolute packed indexes of every changed-token span (bos offset +1 per token)."""
    pieces = [torch.arange(bos_pos + 1 + int(a), bos_pos + 1 + int(b), device=device) for (a, b) in spans]
    if not pieces:
        raise RuntimeError("align: empty changed-token spans")
    return torch.cat(pieces, dim=0)


def _weighted_mean(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """(S, D) hidden, (S,) weights -> mass-normalized (D,) mean via GEMV."""
    mass = weights.sum()
    return (weights @ hidden) / mass


def _cos_rows(rows: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(n, D) rows vs (D,) vector -> (n,) cosines (fp32)."""
    rows = rows.float()
    vector = vector.float()
    return (rows * vector).sum(dim=-1) / (rows.norm(dim=-1) * vector.norm().clamp_min(_EPS) + _EPS)


# ---------------------------------------------------------------------------
# hook manager
# ---------------------------------------------------------------------------
class AlignHookManager:
    """Forward hooks on the selected decoder layers; per-forward configuration.

    Install once on module references captured *before* FSDP/AC wrapping (the
    wrapped tree still calls these exact ``nn.Module`` objects).  Before each
    training forward, ``configure_forward`` supplies the absolute changed-token
    indexes and the in/out weight vectors; the hook immediately reduces the
    (S, D) layer output to (h_tok, mu_in, mu_out) and stashes them with live
    autograd graph.  ``clear`` drops everything after the state's backward.
    """

    def __init__(self, layer_modules: Sequence[nn.Module], layer_indices: Sequence[int]) -> None:
        if len(layer_modules) != len(layer_indices):
            raise ValueError("align: layer modules and indices must be matched (empty = disabled manager)")
        self.layer_modules = list(layer_modules)
        self.layer_indices = [int(i) for i in layer_indices]
        self.handles: List[Any] = []
        self.stash: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._config: Optional[Dict[str, Any]] = None
        self.hook_calls = 0

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("align: hooks already installed")
        for module, index in zip(self.layer_modules, self.layer_indices):
            self.handles.append(module.register_forward_hook(self._make_hook(index)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def configure_forward(self, token_index: torch.Tensor, w_in: torch.Tensor, w_out: torch.Tensor,
                         detach_mu: bool = False) -> None:
        if token_index.numel() == 0:
            raise RuntimeError("align: empty token index")
        if w_in.shape != w_out.shape or w_in.numel() != w_in.numel() or w_in.dim() != 1:
            raise RuntimeError("align: weight vectors must be matching 1-D tensors")
        self.stash = {}
        self._config = {
            "token_index": token_index,
            "w_in": w_in,
            "w_out": w_out,
            "detach_mu": bool(detach_mu),
        }

    def clear(self) -> None:
        self.stash = {}
        self._config = None

    @property
    def configured(self) -> bool:
        return self._config is not None

    def _make_hook(self, layer_index: int):
        def hook(_module: nn.Module, _args: Any, output: Any) -> None:
            self.hook_calls += 1
            config = self._config
            if config is None:
                return
            if layer_index in self.stash:
                return  # first-write-wins: recompute during backward must not swap tensors
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.dim() != 2:
                raise RuntimeError(f"align L{layer_index}: unexpected layer output {type(hidden)}")
            if hidden.shape[0] != config["w_in"].shape[0]:
                raise RuntimeError(
                    f"align L{layer_index}: sequence length {hidden.shape[0]} != weights {config['w_in'].shape[0]}"
                )
            h_tok = hidden.index_select(0, config["token_index"])
            mu_in = _weighted_mean(hidden, config["w_in"])
            mu_out = _weighted_mean(hidden, config["w_out"])
            if config["detach_mu"]:
                mu_in = mu_in.detach()
                mu_out = mu_out.detach()
            self.stash[layer_index] = (h_tok, mu_in, mu_out)

        return hook


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def empty_align_metrics(reason: str = "not_configured") -> Dict[str, Any]:
    return {
        "align_loss": 0.0,
        "align_margin": None,
        "align_cos_in": None,
        "align_cos_out": None,
        "align_tokens": 0,
        "align_layers_used": 0,
        "align_skipped": 1,
        "align_skip_reason": reason,
    }


def align_loss_from_stash(stash: Mapping[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                          layer_indices: Sequence[int],
                          margin_scale: float = 1.0
                          ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """L_align = -log sigmoid(scale * (cos(h_tok, mu_in) - cos(h_tok, mu_out))).

    Averaged over every (changed token, alignment layer) pair of this forward.
    Returns (loss, metrics); loss is None when no configured layer produced an
    extract (caller must then skip the align term instead of adding zero).
    """
    scale = float(margin_scale)
    if scale <= 0.0:
        raise ValueError("align: margin scale must be positive")
    terms: List[torch.Tensor] = []
    margins: List[torch.Tensor] = []
    cos_in_list: List[torch.Tensor] = []
    cos_out_list: List[torch.Tensor] = []
    metrics = empty_align_metrics("no_layer_extract")
    metrics["align_skipped"] = 0
    metrics.pop("align_skip_reason")
    per_layer: Dict[str, float] = {}
    tokens_total = 0
    for index in layer_indices:
        if index not in stash:
            continue
        h_tok, mu_in, mu_out = stash[index]
        cos_in = _cos_rows(h_tok, mu_in)
        cos_out = _cos_rows(h_tok, mu_out)
        margin = scale * (cos_in - cos_out)
        terms.append(-torch.nn.functional.logsigmoid(margin))
        margins.append(margin)
        cos_in_list.append(cos_in)
        cos_out_list.append(cos_out)
        tokens_total += int(h_tok.shape[0])
        per_layer[f"align_margin_L{index}"] = float(margin.detach().float().mean())
    if not terms:
        metrics["align_skipped"] = 1
        metrics["align_skip_reason"] = "no_layer_extract"
        return None, metrics
    loss = torch.cat(terms).mean()
    margin_all = torch.cat(margins)
    metrics.update(
        {
            "align_loss": float(loss.detach().float().cpu()),
            "align_margin": float(margin_all.detach().float().mean().cpu()),
            "align_cos_in": float(torch.cat(cos_in_list).detach().float().mean().cpu()),
            "align_cos_out": float(torch.cat(cos_out_list).detach().float().mean().cpu()),
            "align_tokens": tokens_total,
            "align_layers_used": len(terms),
        }
    )
    metrics.update(per_layer)
    return loss, metrics


# ---------------------------------------------------------------------------
# optional general-T2I replay records (default off)
# ---------------------------------------------------------------------------
def validate_replay_record(record: Mapping[str, Any]) -> Dict[str, str]:
    """Replay JSONL contract: {"pair_id", "source": {"prompt", "cot_text", "image"}}.

    Consumed through the ``standard_cot`` variant, i.e. the base dataset's
    ``_standard_sample(pair, "source")`` path.
    """
    errors: List[str] = []
    pair_id = record.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        errors.append("pair_id must be non-empty text")
    source = record.get("source")
    if not isinstance(source, Mapping):
        errors.append("source must be an object")
    else:
        for field in ("prompt", "image"):
            value = source.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"source.{field} must be non-empty text")
        if not isinstance(source.get("cot_text", ""), str):
            errors.append("source.cot_text must be text (may be empty)")
    if errors:
        raise ValueError(f"replay record invalid: {'; '.join(errors)}")
    return {"pair_id": pair_id, "prompt": source["prompt"], "image": source["image"]}
