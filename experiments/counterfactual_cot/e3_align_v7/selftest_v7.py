#!/usr/bin/env python3
"""CPU selftest for the E3 v7 thrust-fixed alignment trainer (no GPU, no model weights).

Covers the v6 acceptance axes plus the four v7 changes:

1. loss numerical correctness: sigmoid-margin math, monotonicity, weighted
   means, hard/continuous region identities, v5-objective equivalence when the
   anti-overshoot switches are off;
2. TEMPERATURE GRADIENT NUMERICS: at the v6-observed plateau margins
   (m=0.005), tau=0.1 amplifies d(-log sigmoid(m/tau))/dm by ~1/tau = ~10x
   vs the untampered (scale 1.0) gradient; combined with lambda 0.1 -> 0.3
   the effective align-gradient scale-up vs v6 defaults is ~30x;
3. STRENGTH REGULARIZER NUMERICS: L_str = w_s*|E/O.detach() - 1| value and
   gradient directions, the detached-anchor property (outside term has no
   grad), and the live region-tensor stash of the v7 objective;
4. PLATEAU GUARD: pure plateau_check semantics (band escape, trend escape,
   window, alert payload);
plus gradient flow (detach_mu; NON-REENTRANT activation checkpointing +
forward-hook parity — the mechanism train_v7 relies on) and CLI/wiring.

Run:  python3 selftest_v7.py      (exit 0 = all pass)
"""
from __future__ import annotations

import copy
import json
import math
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import torch
from torch import nn

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.e3_align_v6 import align_loss as AL
from experiments.counterfactual_cot.e3_align_v7 import align_objective as OBJ
from experiments.counterfactual_cot.e3_align_v7 import train_v7

V5_MANIFEST = BAGEL_ROOT / "experiments/counterfactual_cot/gold40_automask_v5_data/artifacts/gold40_source_target_automask_v5.jsonl"
LOCAL_LLM = BAGEL_ROOT / "models/BAGEL-7B-MoT"


class MockTokenizer:
    """Char-level fallback so span tests still run without the real tokenizer."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) % 100_000 for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(int(i)) if 0 <= int(i) < 0x10FFFF else "?" for i in ids)


_COUNTING = {"encode": 0}


class CountingTokenizer(MockTokenizer):
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def encode(self, text: str) -> List[int]:
        _COUNTING["encode"] += 1
        return self.inner.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.inner.decode(ids)


def _load_tokenizer() -> Tuple[Any, str]:
    if (LOCAL_LLM / "tokenizer.json").is_file():
        try:
            from experiments.counterfactual_cot.training.smoke_train import _load_tokenizer as _load

            tokenizer, _, _ = _load(str(LOCAL_LLM))
            return tokenizer, "real"
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] real tokenizer load failed ({exc!r}); using mock")
    return MockTokenizer(), "mock"


def _first_manifest_record() -> Dict[str, Any]:
    with V5_MANIFEST.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _render_prompt(record: Dict[str, Any], state_id: str) -> str:
    from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import render_state_prompt

    return render_state_prompt(record, state_id)


# ---------------------------------------------------------------------------
# 1. span locating
# ---------------------------------------------------------------------------
def test_find_changed_occurrences() -> None:
    prompt = "Source CoT: a.object=left_region; b.count=2\nTarget CoT: a.object=right_region; b.count=2"
    occ = AL.find_changed_occurrences(prompt, "right_region")
    assert occ == [(prompt.find("=right_region") + 1, prompt.find("=right_region") + 1 + len("right_region"))], occ
    assert AL.find_changed_occurrences(prompt, "missing_value") == []
    both = AL.find_changed_occurrences(prompt, "b.count")  # not anchored -> plain fallback hits
    assert both, "fallback substring path should hit"


def test_target_cot_occurrences() -> None:
    record = _first_manifest_record()
    for state_id, value_key in (("source_state", "old_value"), ("target_state", "new_value")):
        value = record["intervention"][value_key]
        prompt = _render_prompt(record, state_id)
        spans, source = AL.target_cot_occurrences(prompt, value)
        assert spans, f"{state_id}: value {value!r} not located in prompt"
        assert source == "target_cot", source
        marker = prompt.rfind("Target CoT:")
        for start, _end in spans:
            assert start >= marker + len("Target CoT:"), "occurrence must be inside the Target CoT line"
        # the other state's value must not be picked up in the target prompt's Target CoT line
    other = record["intervention"]["old_value" if record["intervention"].get("old_value") != record["intervention"].get("new_value") else "new_value"]
    prompt = _render_prompt(record, "target_state")
    spans, source = AL.target_cot_occurrences(prompt, other)
    if spans:
        assert source in ("target_cot", "full_prompt")
    else:
        assert source == "none"


def test_state_changed_value() -> None:
    record = _first_manifest_record()
    old_value = record["intervention"].get("old_value")
    new_value = record["intervention"].get("new_value")
    if old_value != new_value:
        assert AL.state_changed_value(record, "source_state") == str(old_value)
        assert AL.state_changed_value(record, "target_state") == str(new_value)
    assert AL.state_changed_value({"intervention": {}}, "target_state") is None


def test_span_to_token_range_real_tokenizer() -> None:
    tokenizer, kind = _load_tokenizer()
    record = _first_manifest_record()
    value = str(record["intervention"]["new_value"])
    prompt = _render_prompt(record, "target_state")
    spans, source = AL.changed_token_spans(tokenizer, prompt, value)
    assert spans and source == "target_cot", (spans, source)
    text_ids = tokenizer.encode(prompt)
    covered = "".join(tokenizer.decode(text_ids[a:b]) for a, b in spans)
    assert value.lower() in covered.lower().replace(" ", ""), (value, covered[:200])
    print(f"    tokenizer={kind} spans={spans} covered={covered.strip()[:60]!r}")


def test_target_cache() -> None:
    tokenizer = CountingTokenizer(MockTokenizer())
    cache = AL.AlignTargetCache(tokenizer, capacity=2)
    prompt = "Target CoT: a.object=right_region"
    n_occurrences = len(AL.find_changed_occurrences(prompt, "right_region"))
    assert n_occurrences >= 1
    _COUNTING["encode"] = 0
    entry1 = cache.entry("p1", "target_state", prompt, "right_region")
    entry2 = cache.entry("p1", "target_state", prompt, "right_region")
    assert entry1 is entry2
    assert entry1["spans"], "span must be located for a cache entry"
    expected_encodes = 1 + 2 * n_occurrences  # prompt encode + start/end prefix encodes per occurrence
    assert _COUNTING["encode"] == expected_encodes, (_COUNTING["encode"], expected_encodes)
    before = _COUNTING["encode"]
    cache.entry("p2", "target_state", prompt, "right_region")  # capacity 2 -> evicts p1
    cache.entry("p3", "target_state", prompt, "right_region")
    assert _COUNTING["encode"] > before, "cache misses must re-encode"


# ---------------------------------------------------------------------------
# 2. packed layout
# ---------------------------------------------------------------------------
def _synthetic_packed(prompt_text_ids: List[int], n_vae: int = 4096, n_vit: int = 4900):
    so_i, eo_i, bos, eos = 151666, 151667, 151643, 151645  # arbitrary distinct ids
    # text channel order: SOI/EOI(src vae) SOI/EOI(vit) bos prompt eos SOI/EOI(tgt vae)
    packed_text_ids = [so_i, eo_i, so_i, eo_i, bos] + list(prompt_text_ids) + [eos, so_i, eo_i]
    vit_last = 4098 + n_vit  # vit tokens occupy 4099..vit_last
    packed_vit = list(range(4099, vit_last + 1))
    text_len = len(prompt_text_ids)
    mse_start = vit_last + text_len + 5  # EOI + bos + eos + SOI + 1 (first latent)
    mse = list(range(mse_start, mse_start + n_vae))
    return packed_text_ids, packed_vit, mse


def test_derive_prompt_layout() -> None:
    prompt_ids = [11, 12, 13, 14, 15, 16]
    packed_text_ids, packed_vit, mse = _synthetic_packed(prompt_ids)
    layout = AL.derive_prompt_layout(packed_text_ids, packed_vit, mse, prompt_ids)
    assert layout["n_text"] == 6, layout
    assert layout["bos_pos"] == 4098 + 4900 + 2, layout  # vit EOI then bos
    assert mse[0] == layout["bos_pos"] + 6 + 2 + 1  # prompt + eos + SOI + 1
    try:
        AL.derive_prompt_layout(packed_text_ids, packed_vit, mse, prompt_ids[:-1])
        raise AssertionError("mismatched prompt ids must raise")
    except RuntimeError:
        pass
    try:
        AL.derive_prompt_layout(packed_text_ids, packed_vit, mse, prompt_ids[:-1] + [99])
        raise AssertionError("different prompt ids must raise")
    except RuntimeError:
        pass
    try:
        AL.derive_prompt_layout(packed_text_ids[:4], packed_vit, mse, prompt_ids)
        raise AssertionError("short packed ids must raise")
    except RuntimeError:
        pass


def test_region_vectors_and_token_index() -> None:
    edit = torch.tensor([0.0, 1.0, 0.5, 0.5])
    outside = 1.0 - edit
    mse_positions = torch.tensor([6, 7, 8, 9])
    w_in, w_out = AL.build_region_weight_vectors(edit, outside, 12, mse_positions, torch.device("cpu"), torch.float32)
    assert w_in.tolist() == [0.0] * 6 + [0.0, 1.0, 0.5, 0.5] + [0.0, 0.0]
    assert torch.allclose(w_in[mse_positions] + w_out[mse_positions], torch.ones(4)), "partition at latent positions"
    assert float(w_in.sum()) == 2.0 and float(w_out.sum()) == 2.0
    outside_mask = torch.ones(12, dtype=torch.bool)
    outside_mask[mse_positions] = False
    assert float(w_in[outside_mask].abs().sum()) == 0.0 and float(w_out[outside_mask].abs().sum()) == 0.0
    index = AL.build_token_index([(1, 2), (4, 5)], bos_pos=100, device=torch.device("cpu"))
    assert index.tolist() == [102, 105], index.tolist()
    try:
        AL.build_token_index([], 100, torch.device("cpu"))
        raise AssertionError("empty spans must raise")
    except RuntimeError:
        pass


def test_weighted_mean() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(64, 8)
    weights = torch.rand(64)
    reference = (hidden * weights[:, None]).sum(0) / weights.sum()
    got = AL._weighted_mean(hidden, weights)
    assert torch.allclose(got, reference, atol=1e-6)
    got_bf16 = AL._weighted_mean(hidden.bfloat16(), weights.bfloat16()).float()
    assert torch.allclose(got_bf16, reference, atol=5e-2), "bf16 GEMV should stay within 5% of fp32 reference"


# ---------------------------------------------------------------------------
# 3. loss math
# ---------------------------------------------------------------------------
def _make_stash(sign: float, dtype: torch.dtype = torch.float32, requires_grad: bool = True):
    """Deterministic construction: rows are distinct unit basis vectors e_0..e_4
    (dim 8); mu_in is +/- the all-ones direction (cos = +/- 1/sqrt(8) per row);
    mu_out is e_7 (orthogonal to every row, cos = 0).  Margin per row is
    therefore exactly sign / sqrt(8) before margin_scale."""
    rows = torch.zeros(5, 8, dtype=dtype)
    for i in range(5):
        rows[i, i] = 1.0
    mu_in = torch.full((8,), sign / math.sqrt(8.0), dtype=dtype)
    mu_out = torch.zeros(8, dtype=dtype)
    mu_out[7] = 1.0
    if requires_grad:
        rows.requires_grad_(True)
        mu_in.requires_grad_(True)
        mu_out.requires_grad_(True)
    return rows, mu_in, mu_out


def test_align_loss_math() -> None:
    unit = 1.0 / math.sqrt(8.0)
    # exact identity: loss == -log sigmoid(margin), margin == sign/sqrt(8)
    rows, mu_in, mu_out = _make_stash(+1.0)
    loss, metrics = AL.align_loss_from_stash({13: (rows, mu_in, mu_out)}, [13], 1.0)
    assert math.isclose(metrics["align_margin"], unit, rel_tol=1e-5), metrics["align_margin"]
    assert math.isclose(loss.item(), -math.log(math.exp(unit) / (1 + math.exp(unit))), rel_tol=1e-5)
    assert math.isclose(metrics["align_cos_in"], unit, rel_tol=1e-5)
    assert math.isclose(metrics["align_cos_out"], 0.0, abs_tol=1e-6)

    # sign ordering + bounds
    _, metrics_positive = AL.align_loss_from_stash({6: _make_stash(+1.0)}, [6], 1.0)
    _, metrics_negative = AL.align_loss_from_stash({6: _make_stash(-1.0)}, [6], 1.0)
    assert metrics_positive["align_margin"] > 0 > metrics_negative["align_margin"]
    assert 0.0 < metrics_positive["align_loss"] < math.log(2.0) < metrics_negative["align_loss"]

    # margin scale = inverse temperature
    _, metrics_scaled = AL.align_loss_from_stash({6: _make_stash(+1.0)}, [6], 5.0)
    assert math.isclose(metrics_scaled["align_margin"], 5.0 * unit, rel_tol=1e-5)
    assert metrics_scaled["align_loss"] < metrics_positive["align_loss"]

    # bf16 stash tolerated
    loss_bf16, metrics_bf16 = AL.align_loss_from_stash({16: _make_stash(+1.0, dtype=torch.bfloat16)}, [16], 1.0)
    assert torch.isfinite(loss_bf16) and metrics_bf16["align_skipped"] == 0
    assert abs(metrics_bf16["align_margin"] - unit) < 1e-2

    # multi-layer averaging
    loss_multi, metrics_multi = AL.align_loss_from_stash(
        {13: _make_stash(+1.0), 6: _make_stash(+1.0), 16: _make_stash(-1.0), 14: _make_stash(+1.0)},
        [13, 6, 16, 14],
        1.0,
    )
    assert metrics_multi["align_layers_used"] == 4
    assert all(f"align_margin_L{index}" in metrics_multi for index in (13, 6, 16, 14))
    expected = sum(metrics_multi[f"align_margin_L{index}"] for index in (13, 6, 16, 14)) / 4
    assert math.isclose(metrics_multi["align_margin"], expected, rel_tol=1e-5)

    # skip paths
    loss_none, metrics_none = AL.align_loss_from_stash({}, [13], 1.0)
    assert loss_none is None and metrics_none["align_skipped"] == 1
    empty = AL.empty_align_metrics("lambda_zero")
    assert empty["align_loss"] == 0.0 and empty["align_skip_reason"] == "lambda_zero"
    try:
        AL.align_loss_from_stash({13: (rows, mu_in, mu_out)}, [13], 0.0)
        raise AssertionError("non-positive scale must raise")
    except ValueError:
        pass


def test_align_grad_flow_and_detach() -> None:
    rows, mu_in, mu_out = _make_stash(+1.0)
    loss, _ = AL.align_loss_from_stash({13: (rows, mu_in, mu_out)}, [13], 1.0)
    loss.backward()
    assert rows.grad is not None and torch.isfinite(rows.grad).all()
    assert mu_in.grad is not None and mu_out.grad is not None
    assert torch.isfinite(mu_in.grad).all() and torch.isfinite(mu_out.grad).all()
    # detached mu: gradient still reaches the token side
    rows2, mu_in2, mu_out2 = _make_stash(+1.0)
    mu_in2, mu_out2 = mu_in2.detach(), mu_out2.detach()
    loss2, _ = AL.align_loss_from_stash({13: (rows2, mu_in2, mu_out2)}, [13], 1.0)
    loss2.backward()
    assert rows2.grad is not None and torch.isfinite(rows2.grad).all()
    assert torch.allclose(rows2.grad, rows.grad, atol=1e-6), "detached mu must not change token-side grads"


# ---------------------------------------------------------------------------
# 4. activation-checkpointing parity (the mechanism train_v7 depends on)
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _Stack(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _ac_parity_inner() -> None:
    from functools import partial

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    torch.manual_seed(0)
    dim, seq_len = 8, 16
    baseline = _Stack(dim, 3)
    wrapped = copy.deepcopy(baseline)
    apply_activation_checkpointing(
        wrapped,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, _Block),
    )
    n_wrapped = sum(1 for module in wrapped.modules() if type(module).__name__ == "CheckpointWrapper")
    assert n_wrapped == 3, f"expected 3 checkpoint wrappers, got {n_wrapped}"
    assert sum(1 for module in wrapped.modules() if isinstance(module, _Block)) == 3

    token_index = torch.tensor([2, 5, 7])
    weights_in = torch.zeros(seq_len)
    weights_in[10:] = torch.rand(6)
    weights_out = torch.zeros(seq_len)
    weights_out[:5] = torch.rand(5)
    assert weights_in.sum() > 0 and weights_out.sum() > 0

    hooks = AL.AlignHookManager([wrapped.blocks[1]], [1])
    hooks.install()
    hooks.configure_forward(token_index, weights_in, weights_out, detach_mu=False)

    x = torch.randn(seq_len, dim, requires_grad=True)
    out_wrapped = wrapped(x)
    assert hooks.configured and 1 in hooks.stash
    h_tok, mu_in, mu_out = hooks.stash[1]
    assert h_tok.requires_grad and mu_in.requires_grad and mu_out.requires_grad, \
        "hook tensors must stay autograd-connected under NON_REENTRANT activation checkpointing"
    assert h_tok.shape == (3, dim)
    stash_id = id(h_tok)
    sft_wrapped = out_wrapped.square().mean()
    align_wrapped, metrics = AL.align_loss_from_stash(hooks.stash, [1], 1.0)
    assert align_wrapped is not None and metrics["align_tokens"] == 3
    (sft_wrapped + 0.1 * align_wrapped).backward()
    assert id(hooks.stash[1][0]) == stash_id, "recompute during backward must not swap stashed tensors"

    # baseline (no checkpointing) reproduces identical gradients
    x_base = x.detach().clone().requires_grad_(True)
    hidden = baseline.blocks[0](x_base)
    mid = baseline.blocks[1](hidden)
    reference_h_tok = mid.index_select(0, token_index)
    reference_mu_in = AL._weighted_mean(mid, weights_in)
    reference_mu_out = AL._weighted_mean(mid, weights_out)
    ref_stash = {1: (reference_h_tok, reference_mu_in, reference_mu_out)}
    ref_align, _ = AL.align_loss_from_stash(ref_stash, [1], 1.0)
    out_baseline = baseline.blocks[2](mid)
    (out_baseline.square().mean() + 0.1 * ref_align).backward()

    assert torch.allclose(x_base.grad, x.grad, atol=1e-5), "input grads diverge between AC and baseline"
    for (name_a, param_a), (name_b, param_b) in zip(baseline.named_parameters(), wrapped.named_parameters()):
        assert name_a == _strip_wrapper_names(name_b), (name_a, name_b)
        assert param_a.grad is not None and param_b.grad is not None, name_a
        assert torch.allclose(param_a.grad, param_b.grad, atol=1e-5), f"grad mismatch at {name_a}"
    print("    AC parity: input + all block params match non-checkpointed baseline (atol 1e-5)")


def _strip_wrapper_names(name: str) -> str:
    for fragment in ("_checkpoint_wrapped_module.", "_fsdp_wrapped_module."):
        name = name.replace(fragment, "")
    return name.replace("..", ".").strip(".")


def test_hook_noop_when_unconfigured() -> None:
    torch.manual_seed(1)
    stack = _Stack(8, 2)
    hooks = AL.AlignHookManager([stack.blocks[1]], [1])
    hooks.install()
    out = stack(torch.randn(16, 8))
    out.sum().backward()
    assert hooks.stash == {} and hooks.hook_calls == 1, "unconfigured hooks must stash nothing"
    hooks.remove()
    out = stack(torch.randn(16, 8))
    assert hooks.hook_calls == 1, "removed hooks must not fire"


# ---------------------------------------------------------------------------
# 5. objective module
# ---------------------------------------------------------------------------
class _MockModel(nn.Module):
    def __init__(self, mse: torch.Tensor, ce: torch.Tensor) -> None:
        super().__init__()
        self._mse = mse
        self._ce = ce

    def forward(self, **data: Any) -> Dict[str, torch.Tensor]:
        return {"mse": self._mse, "ce": self._ce}


def _craft_weights() -> Dict[str, torch.Tensor]:
    torch.manual_seed(11)
    edit = torch.zeros(4096)
    edit[100:160] = torch.linspace(0.2, 1.0, 60)  # a soft mask blob
    edit[200:220] = 1.0
    outside = 1.0 - edit
    return {"edit": edit, "outside": outside}


def test_objective_matches_v5_when_switches_off() -> None:
    from experiments.counterfactual_cot.gold40_automask_v5_data.automask_objective import automask_forward_loss as v5_forward

    torch.manual_seed(5)
    mse = torch.rand(4096, 64)
    ce = torch.rand(())
    weights = _craft_weights()
    model = _MockModel(mse, ce)
    v5_total, v5_metrics = v5_forward(model, {}, weights, lambda _m: nullcontext())
    OBJ.configure_overshoot(edit_weight=0.5, debias=False)
    v6_total, v6_metrics = OBJ.paired_automask_forward_loss(model, {}, "target_state", {"source_state": weights, "target_state": weights}, lambda _m: nullcontext())
    assert math.isclose(v5_total.item(), v6_total.item(), rel_tol=1e-6), (v5_total.item(), v6_total.item())
    for key in ("edit_mse", "outside_mse", "native_mse", "ce", "total"):
        assert math.isclose(v5_metrics[key], v6_metrics[key], rel_tol=1e-6), key
    assert v6_metrics["region_mode"] == "continuous" and v6_metrics["edit_weight"] == 0.5


def test_objective_debias_and_rebalance() -> None:
    torch.manual_seed(6)
    mse = torch.rand(4096, 64)
    ce = torch.rand(())
    weights = _craft_weights()
    per_token = mse.mean(-1)
    hard_in = weights["edit"] >= OBJ.HARD_IN_THRESHOLD
    n_in = int(hard_in.sum())
    model = _MockModel(mse, ce)
    OBJ.configure_overshoot(edit_weight=0.4, debias=True)
    total, metrics = OBJ.paired_automask_forward_loss(model, {}, "target_state", {"source_state": weights, "target_state": weights}, lambda _m: nullcontext())
    assert metrics["region_mode"] == "hard_partition"
    assert metrics["edit_weight"] == 0.4
    expected_edit = per_token[hard_in].mean().item()
    expected_outside = per_token[~hard_in].mean().item()
    assert math.isclose(metrics["edit_mse"], expected_edit, rel_tol=1e-6)
    assert math.isclose(metrics["outside_mse"], expected_outside, rel_tol=1e-6)
    assert math.isclose(metrics["region_mse"], 0.4 * expected_edit + 0.6 * expected_outside, rel_tol=1e-6)
    # exact count identity backs the internal recomposition check
    native = per_token.mean().item()
    assert math.isclose(native, (n_in * expected_edit + (4096 - n_in) * expected_outside) / 4096, rel_tol=1e-6)
    assert n_in > 0


def test_objective_debias_fallback_and_validation() -> None:
    torch.manual_seed(7)
    mse = torch.rand(4096, 64)
    weights = {"edit": torch.full((4096,), 0.9), "outside": torch.full((4096,), 0.1)}
    model = _MockModel(mse, torch.rand(()))
    OBJ.configure_overshoot(edit_weight=0.4, debias=True)
    _, metrics = OBJ.paired_automask_forward_loss(model, {}, "target_state", {"source_state": weights, "target_state": weights}, lambda _m: nullcontext())
    assert metrics["region_mode"] == "debias_fallback_continuous", "empty hard outside region must fall back"
    for bad in (0.0, 1.0, -0.2, 1.5):
        try:
            OBJ.configure_overshoot(edit_weight=bad, debias=False)
            raise AssertionError(f"edit_weight={bad} must raise")
        except Exception as exc:  # noqa: BLE001
            assert "edit_weight" in str(exc)
    OBJ.configure_overshoot(edit_weight=0.4, debias=False)
    try:
        OBJ.paired_automask_forward_loss(model, {}, "unknown_state", {"target_state": weights}, lambda _m: nullcontext())
        raise AssertionError("unknown branch must raise")
    except Exception:  # noqa: BLE001
        pass


def test_objective_build_matches_v5_partition() -> None:
    from experiments.counterfactual_cot.gold40_automask_v5_data.automask_objective import build_paired_automask_objective as v5_build

    record = _first_manifest_record()
    spec, branch_weights = OBJ.build_paired_automask_objective(record)
    spec_ref, branch_ref = v5_build(record)
    assert spec["objective_id"] == spec_ref["objective_id"]
    assert set(branch_weights) == set(branch_ref) == {"source_state", "target_state"}
    assert branch_weights["source_state"]["edit"] is branch_weights["target_state"]["edit"]
    assert torch.equal(branch_weights["target_state"]["edit"], branch_ref["target_state"]["edit"])
    assert torch.equal(branch_weights["target_state"]["outside"], branch_ref["target_state"]["outside"])
    assert spec["token_grid_hw"] == [64, 64]
    edit_mass = float(branch_weights["target_state"]["edit"].sum())
    print(f"    partition OK: edit_mass={edit_mass:.1f} tokens of 4096")


# ---------------------------------------------------------------------------
# 6. replay records + CLI
# ---------------------------------------------------------------------------
def test_validate_replay_record() -> None:
    ok = {"pair_id": "g1", "source": {"prompt": "a cat", "cot_text": "", "image": "/tmp/x.png"}}
    assert OBJ is not None and AL.validate_replay_record(ok)["pair_id"] == "g1"
    for bad in (
        {"pair_id": "", "source": {"prompt": "a", "cot_text": "", "image": "x"}},
        {"pair_id": "g", "source": {"prompt": "a", "cot_text": "", "image": ""}},
        {"pair_id": "g", "source": {"cot_text": "", "image": "x"}},
        {"pair_id": "g"},
    ):
        try:
            AL.validate_replay_record(bad)
            raise AssertionError(f"bad record must raise: {bad}")
        except ValueError:
            pass


def test_cli_defaults_and_guards() -> None:
    base = ["train_v7.py", "--activation-checkpointing"]  # mandatory for any real (non-preflight) run
    with mock.patch.object(sys, "argv", base):
        args = train_v7._parse_args()
    # v7 thrust defaults
    assert args.align_lambda == 0.3, args.align_lambda
    assert args.align_tau == 0.1 and args.effective_margin_scale == 10.0
    assert args.align_margin_scale == 10.0, "args.align_margin_scale must carry the effective scale"
    assert args.align_layers_list == [13, 6, 16, 14]
    assert args.align_detach_mu is False
    # strength regularizer defaults (replaces the v6 edit-weight rebalance)
    assert args.strength_reg == "on" and args.strength_reg_on is True
    assert args.strength_weight == 0.05 and args.strength_target == 1.0
    assert args.effective_edit_weight == 0.5, "strength-reg on forces v5 0.5/0.5 region weights"
    assert args.effective_mask_debias is True
    # plateau watchdog defaults
    assert args.plateau_guard == "on" and args.plateau_guard_on is True
    assert args.plateau_window == 1000 and args.plateau_center == 0.69
    assert args.plateau_eps == 0.005 and args.plateau_trend_eps == 1e-4
    assert str(args.progress_file).endswith("wb_e3_progress_v7.txt")
    assert args.general_replay_manifest is None and args.general_replay_every == 4
    assert args.init_trainable_from is None
    assert args.steps == 2973 and args.checkpoint_every == 200
    with mock.patch.object(sys, "argv", ["train_v7.py", "--preflight-only"]):
        args_preflight = train_v7._parse_args()
    assert args_preflight.align_lambda == 0.3  # preflight-only may omit the AC flag
    # explicit margin-scale ablation overrides 1/tau
    with mock.patch.object(sys, "argv", base + ["--align-margin-scale", "2.0"]):
        assert train_v7._parse_args().effective_margin_scale == 2.0
    # strength-reg off restores the v6-style rebalance path
    with mock.patch.object(sys, "argv", base + ["--strength-reg", "off", "--edit-loss-weight", "0.4"]):
        args_v6_style = train_v7._parse_args()
    assert args_v6_style.strength_reg_on is False and args_v6_style.effective_edit_weight == 0.4
    with mock.patch.object(sys, "argv", base + ["--overshoot-fix", "off"]):
        args_off = train_v7._parse_args()
    assert args_off.effective_edit_weight == 0.5 and args_off.effective_mask_debias is False
    with mock.patch.object(sys, "argv", base + ["--align-layers", "5"]):
        assert train_v7._parse_args().align_layers_list == [5]
    with mock.patch.object(sys, "argv", base + ["--align-lambda", "0"]):
        assert train_v7._parse_args().align_lambda == 0.0  # lambda=0 disables alignment entirely
    for bad_argv in (
        ["--align-layers", "5,5"],
        ["--align-layers", "99"],
        ["--align-layers", ""],
        ["--align-lambda", "-0.1"],
        ["--align-tau", "0"],
        ["--align-tau", "-0.5"],
        ["--align-margin-scale", "0"],
        ["--edit-loss-weight", "1.0"],
        ["--edit-loss-weight", "0.4"],  # rejected: strength-reg on replaces the rebalance
        ["--strength-weight", "-0.01"],
        ["--strength-target", "0"],
        ["--plateau-window", "0"],
        ["--plateau-eps", "0"],
        ["--plateau-trend-eps", "-1"],
        ["--general-replay-every", "0"],
    ):
        try:
            with mock.patch.object(sys, "argv", base + bad_argv):
                train_v7._parse_args()
            raise AssertionError(f"bad args must exit: {bad_argv}")
        except SystemExit:
            pass
    try:
        with mock.patch.object(sys, "argv", ["train_v7.py"]):  # missing mandatory AC flag
            train_v7._parse_args()
        raise AssertionError("missing --activation-checkpointing must exit")
    except SystemExit:
        pass


def test_resume_guard_keys_cover_align_config() -> None:
    for key in ("align_lambda", "align_layers", "align_margin_scale", "align_tau", "align_detach_mu",
                "strength_reg", "strength_weight", "strength_target", "plateau_guard",
                "overshoot_edit_weight", "overshoot_mask_debias", "general_replay_manifest",
                "general_replay_every", "init_trainable_from"):
        assert key in train_v7.RESUME_GUARD_KEYS, key


# ---------------------------------------------------------------------------
# 7. integration: v6-style combined step on a toy stack
# ---------------------------------------------------------------------------
def test_combined_step_metrics_and_grads() -> None:
    from functools import partial

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    torch.manual_seed(9)
    seq_len, dim = 24, 8
    stack = _Stack(dim, 3)
    apply_activation_checkpointing(
        stack,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, _Block),
    )
    token_index = torch.tensor([3, 4, 11])
    w_in = torch.zeros(seq_len)
    w_in[16:24] = torch.rand(8)
    w_out = torch.zeros(seq_len)
    w_out[0:8] = torch.rand(8)
    hooks = AL.AlignHookManager([stack.blocks[2]], [2])
    hooks.install()
    hooks.configure_forward(token_index, w_in, w_out, detach_mu=False)
    data = {"sequence_length": seq_len}
    x = torch.randn(seq_len, dim, requires_grad=True)

    def forward_loss(model, batch_data, state_id, branch_weights, autocast):
        with autocast(model):
            out = model(**batch_data)
        sft = out.square().mean()
        return sft, {"total": float(sft.detach())}

    class _Ctx:
        def __call__(self, _model):
            return nullcontext()

    sft_loss, metrics = forward_loss(stack, {"x": x}, "target_state", {}, _Ctx())
    align_loss, align_metrics = AL.align_loss_from_stash(hooks.stash, [2], 1.0)
    assert align_loss is not None
    for key in ("align_loss", "align_margin", "align_cos_in", "align_cos_out", "align_tokens", "align_layers_used", "align_margin_L2"):
        assert key in align_metrics, key
    total = sft_loss + 0.3 * align_loss  # v7 thrust lambda
    total.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in stack.named_parameters():
        assert param.grad is not None, f"missing grad: {name}"
    hooks.clear()
    assert hooks.stash == {} and not hooks.configured
    row = {"target_state_align_loss": align_metrics["align_loss"]}
    json.dumps(row)  # history rows must serialize
    print(f"    combined step: sft={float(sft_loss):.4f} align={align_metrics['align_loss']:.4f} margin={align_metrics['align_margin']:+.4f}")


# ---------------------------------------------------------------------------
# 8. v7 thrust fixes: temperature gradient, strength regularizer, plateau guard
# ---------------------------------------------------------------------------
def test_temperature_gradient_magnitude() -> None:
    """tau=0.1 vs tau=1 at the v6-observed margin m=0.005.

    d(-log sigmoid(m/tau))/dm = -(1/tau) * sigmoid(-m/tau); at small |m| the
    amplification vs the untampered gradient (1-softmax terms equal ~0.5) is
    ~1/tau = 10x.  Combined with lambda 0.1 -> 0.3 the total-loss align
    gradient scale-up vs v6 defaults is ~3 * 10 = ~30x.
    """
    margin = 0.005
    m = torch.tensor(margin, requires_grad=True)
    loss_tau = -torch.nn.functional.logsigmoid(m / 0.1)
    loss_tau.backward()
    grad_tau = abs(float(m.grad))
    m2 = torch.tensor(margin, requires_grad=True)
    loss_plain = -torch.nn.functional.logsigmoid(m2 / 1.0)
    loss_plain.backward()
    grad_plain = abs(float(m2.grad))
    expected_tau = (1 / 0.1) * torch.sigmoid(torch.tensor(-margin / 0.1)).item()
    expected_plain = torch.sigmoid(torch.tensor(-margin)).item()
    assert math.isclose(grad_tau, expected_tau, rel_tol=1e-6), (grad_tau, expected_tau)
    assert math.isclose(grad_plain, expected_plain, rel_tol=1e-6), (grad_plain, expected_plain)
    ratio = grad_tau / grad_plain
    assert 9.0 <= ratio <= 10.5, f"temperature amplification should be ~1/tau=10x, got {ratio:.3f}"
    # lambda-scaled total-loss gradient vs v6 defaults: 0.3*grad_tau vs 0.1*grad_plain
    combined = (0.3 * grad_tau) / (0.1 * grad_plain)
    assert 28.0 <= combined <= 32.0, f"expected ~30x (3*lambda * 10x temp), got {combined:.2f}"
    print(f"    grad(m=0.005): tau=0.1 -> {grad_tau:.4f}, no temp -> {grad_plain:.4f} "
          f"(x{ratio:.2f}); with lambda 0.3 vs v6 lambda 0.1: x{combined:.1f}")

    # real-path check through align_loss_from_stash: same amplification on the
    # mu_in gradient when margin_scale = 1/tau instead of 1.0.
    # Construction: rows are unit basis vectors e_0..e_4; mu_in is a UNIT vector
    # with mu_in[i] = 0.005 for i in 0..4 (cos_in = 0.005 per row) and the norm
    # carried by the last 3 components; mu_out = e_7 is orthogonal to every row
    # (cos_out = 0) -> margin is exactly 0.005 before margin_scale.
    small = 0.005
    tail = math.sqrt((1.0 - 5.0 * small * small) / 3.0)
    rows = torch.zeros(5, 8)
    for i in range(5):
        rows[i, i] = 1.0
    mu_out = torch.zeros(8)
    mu_out[7] = 1.0

    def _mu_in_grad(scale: float) -> float:
        rows_leaf = rows.clone().requires_grad_(True)
        mu_in = torch.tensor([small] * 5 + [tail] * 3, requires_grad=True)
        loss, metrics = AL.align_loss_from_stash({13: (rows_leaf, mu_in, mu_out)}, [13], scale)
        assert math.isclose(metrics["align_margin"], scale * small, rel_tol=1e-6), metrics["align_margin"]
        assert math.isclose(metrics["align_cos_out"], 0.0, abs_tol=1e-6)
        loss.backward()
        return float(mu_in.grad.abs().sum())

    grad_scale10 = _mu_in_grad(10.0)
    grad_scale1 = _mu_in_grad(1.0)
    ratio_real = grad_scale10 / grad_scale1
    assert 9.0 <= ratio_real <= 10.5, f"real-path amplification {ratio_real:.3f} should be ~10x"


def test_strength_reg_math() -> None:
    """L_str = w_s * |E/O.detach() - target|: values, signs, detached anchor."""
    w_s, target = 0.05, 1.0
    # under-edit regime (v6 observed E/O ~ 2): pushes E down, O untouched
    edit = torch.tensor(0.6, requires_grad=True)
    outside = torch.tensor(0.3, requires_grad=True)
    strength = edit / outside.detach()
    loss = w_s * torch.abs(strength - target)
    assert math.isclose(float(loss), 0.05 * (0.6 / 0.3 - 1.0), rel_tol=1e-6)
    loss.backward()
    assert edit.grad is not None and float(edit.grad) > 0, "under-edit: gradient must reduce edit_mse"
    assert outside.grad is None, "detached anchor: no gradient may reach the outside region"
    # over-fit side (E < O): two-sided pull back up (bounded by w_s; SFT dominates)
    edit2 = torch.tensor(0.3, requires_grad=True)
    outside2 = torch.tensor(0.6, requires_grad=True)
    loss2 = w_s * torch.abs(edit2 / outside2.detach() - target)
    assert math.isclose(float(loss2), 0.05 * 0.5, rel_tol=1e-6)
    loss2.backward()
    assert float(edit2.grad) < 0, "over side resists further edit-region reduction"


def test_strength_reg_real_path() -> None:
    """The v7 objective exposes live region tensors; the trainer formula agrees."""
    torch.manual_seed(12)
    mse = torch.rand(4096, 64, requires_grad=True)
    ce = torch.rand(())
    weights = _craft_weights()
    model = _MockModel(mse, ce)
    OBJ.configure_overshoot(edit_weight=0.5, debias=False)
    OBJ.clear_branch_tensors()
    total, metrics = OBJ.paired_automask_forward_loss(
        model, {}, "target_state", {"source_state": weights, "target_state": weights},
        lambda _m: nullcontext(),
    )
    extracted = OBJ.branch_region_tensors("target_state")
    assert extracted is not None, "forward must stash the live region tensors"
    edit_region, outside_region = extracted
    assert edit_region.requires_grad and outside_region.requires_grad, "stash must stay autograd-connected"
    strength = edit_region / outside_region.detach().clamp_min(1e-6)
    loss_str = 0.05 * torch.abs(strength - 1.0)
    assert math.isclose(
        float(loss_str.detach()),
        0.05 * abs(metrics["edit_mse"] / metrics["outside_mse"] - 1.0),
        rel_tol=1e-5,
    ), "regularizer value must match the logged edit/outside statistics"
    (total + loss_str).backward()
    assert mse.grad is not None and torch.isfinite(mse.grad).all(), "L_str gradient must reach the mse leaf"
    # stash hygiene: clearing drops the entries; unknown branches return None
    OBJ.clear_branch_tensors()
    assert OBJ.branch_region_tensors("target_state") is None
    assert OBJ.branch_region_tensors("source_state") is None, "only requested branches are stashed"


def test_plateau_guard_semantics() -> None:
    check = train_v7.plateau_check
    window = [0.6920 + 0.0001 * (i % 7) for i in range(1000)]  # all inside [0.685, 0.695]
    alert = check(window, center=0.69, eps=0.005, trend_eps=1e-4)
    assert alert is not None and alert["window"] == 1000
    assert math.isclose(alert["band"][0], 0.685) and math.isclose(alert["band"][1], 0.695)
    assert alert["values_min"] >= 0.685 and alert["values_max"] <= 0.695
    # any escape from the band -> no alert (something moved)
    escaped = list(window)
    escaped[500] = 0.5
    assert check(escaped, 0.69, 0.005, 1e-4) is None
    escaped_hi = list(window)
    escaped_hi[500] = 0.696  # just above the upper band edge
    assert check(escaped_hi, 0.69, 0.005, 1e-4) is None
    # downward trend within the band -> no alert (leaving the plateau)
    declining = [0.6945 - 0.000009 * i for i in range(1000)]  # all in band, drop ~0.009
    assert check(declining, 0.69, 0.005, 1e-4) is None
    # noise-level drift (< trend_eps) still counts as stuck
    drift = [0.6900 - 0.00000004 * i for i in range(1000)]  # drop 4e-5 < 1e-4
    assert check(drift, 0.69, 0.005, 1e-4) is not None
    # too few values -> no alert
    assert check([0.692], 0.69, 0.005, 1e-4) is None
    assert check([], 0.69, 0.005, 1e-4) is None


TESTS = [
    ("find_changed_occurrences", test_find_changed_occurrences),
    ("target_cot_occurrences", test_target_cot_occurrences),
    ("state_changed_value", test_state_changed_value),
    ("span_to_token_range (real tokenizer)", test_span_to_token_range_real_tokenizer),
    ("AlignTargetCache", test_target_cache),
    ("derive_prompt_layout", test_derive_prompt_layout),
    ("region vectors + token index", test_region_vectors_and_token_index),
    ("_weighted_mean fp32/bf16", test_weighted_mean),
    ("align loss math", test_align_loss_math),
    ("align grad flow + detach_mu", test_align_grad_flow_and_detach),
    ("AC parity (non-reentrant + hooks)", _ac_parity_inner),
    ("hook no-op when unconfigured", test_hook_noop_when_unconfigured),
    ("objective == v5 (switches off)", test_objective_matches_v5_when_switches_off),
    ("objective debias + rebalance", test_objective_debias_and_rebalance),
    ("objective debias fallback + validation", test_objective_debias_fallback_and_validation),
    ("objective partition == v5", test_objective_build_matches_v5_partition),
    ("validate_replay_record", test_validate_replay_record),
    ("CLI defaults + guards", test_cli_defaults_and_guards),
    ("resume guard keys", test_resume_guard_keys_cover_align_config),
    ("combined step metrics + grads", test_combined_step_metrics_and_grads),
    ("temperature gradient magnitude (tau vs none)", test_temperature_gradient_magnitude),
    ("strength reg math (value + detach anchor)", test_strength_reg_math),
    ("strength reg real path (objective stash)", test_strength_reg_real_path),
    ("plateau guard semantics", test_plateau_guard_semantics),
]


def main() -> None:
    failures = 0
    for index, (name, func) in enumerate(TESTS, 1):
        try:
            func()
            print(f"PASS {index:02d}/{len(TESTS)} {name}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {index:02d}/{len(TESTS)} {name}")
            traceback.print_exc()
    print(f"{'ALL PASS' if failures == 0 else f'{failures} FAILURES'} ({len(TESTS)} tests)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
