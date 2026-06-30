"""Tests for the compiled-lexicon modules: the registry, the soft-traversal
module's contract and semantics, and its integration into the lane model."""

import pytest
import torch
from scribblez.max_move_per_lane.lexicon_compiler import N_LETTERS, CompiledLexicon
from scribblez.max_move_per_lane.lexicon_modules import (
    KvMemoryLexicon,
    OracleCrosscheckLexicon,
    SoftTraversalLexicon,
    StraightThroughLexicon,
    available_modules,
    build_lexicon_module,
    parse_module_opts,
    resolve_lane_ffn_mult,
)
from scribblez.max_move_per_lane.model import MaxMovePerLaneModel, compute_loss

WORDS = ["CAT", "CAR", "CARE", "CARES", "CATS", "AT"]


def _lexicon():
    return CompiledLexicon.from_words(WORDS)


def _board(word, gaps=()):
    """One-hot (1, 15, 26) board letters; cells in `gaps` are left empty."""
    t = torch.zeros(1, 15, N_LETTERS)
    for i, ch in enumerate(word):
        if i not in gaps:
            t[0, i, ord(ch) - ord("A")] = 1.0
    return t


def test_registry_and_build():
    assert "none" in available_modules()
    assert {"soft_traversal", "straight_through", "oracle_crosscheck", "kv_memory"} <= set(
        available_modules()
    )
    assert build_lexicon_module("none", channels=8, kwg_path="/nonexistent") is None
    with pytest.raises(KeyError):
        build_lexicon_module("bogus", channels=8, kwg_path="/nonexistent")


def test_straight_through_is_exact_and_differentiable():
    mod = StraightThroughLexicon(channels=16, compiled=_lexicon(), topk=8)

    # Forward commits to one path: acceptance is a crisp 0/1 fact. CAT is a word;
    # CBT is not -- and there is no top-K smear in between.
    feats = torch.zeros(1, 15, 16)
    _force_query(mod, "A")
    assert mod(feats, _board("CAT", gaps=(1,))).cell_signals[0, 2, 0].item() == pytest.approx(1.0)
    _force_query(mod, "B")
    assert mod(feats, _board("CAT", gaps=(1,))).cell_signals[0, 2, 0].item() == pytest.approx(0.0)

    # Straight-through still routes gradient to the query head (nonzero features).
    mod2 = StraightThroughLexicon(channels=16, compiled=_lexicon(), topk=8)
    rfeats = torch.randn(2, 15, 16, requires_grad=True)
    mod2(rfeats, torch.zeros(2, 15, N_LETTERS)).cell_residual.sum().backward()
    assert mod2.query.weight.grad.abs().sum() > 0


def _model_with(module):
    return MaxMovePerLaneModel(
        31, 27, trunk_channels=8, num_blocks=1, lane_layers=1, lexicon_module=module
    )


def test_oracle_crosscheck_signals_and_integration():
    orc = OracleCrosscheckLexicon(channels=8, compiled=_lexicon())
    out = orc(torch.zeros(1, 15, 8), _board("CA"))
    assert out.cell_residual.shape == (1, 15, 8)
    assert out.tokens.shape == (1, 2, 8)
    assert out.cell_signals.shape == (1, 15, 1 + N_LETTERS + N_LETTERS)

    sig = out.cell_signals[0]
    run, cont, accept = sig[:, 0], sig[:, 1 : 1 + N_LETTERS], sig[:, 1 + N_LETTERS :]
    r, t, e = (ord(x) - ord("A") for x in "RTE")
    # After the run "CA": R and T extend it (CAR, CAT) and both complete words;
    # "CA" itself is not a word, and E does not extend it.
    assert run[2] == pytest.approx(0.0)
    assert cont[2, r] > 0.5 and cont[2, t] > 0.5 and cont[2, e] < 0.5
    assert accept[2, r] > 0.5 and accept[2, t] > 0.5
    # On a fully-placed "CAT", the run is a complete word at the following cell.
    assert orc(torch.zeros(1, 15, 8), _board("CAT")).cell_signals[0, 3, 0] == pytest.approx(1.0)

    # A cheat: no learnable query head; only the readout trains.
    assert not hasattr(orc, "query")
    orc(torch.zeros(1, 15, 8), _board("CAT")).cell_residual.sum().backward()
    assert orc.readout.weight.grad.abs().sum() > 0

    out = _model_with(OracleCrosscheckLexicon(channels=8, compiled=_lexicon()))(
        torch.zeros(2, 31, 15, 15), torch.rand(2, 27)
    )
    assert out["lane_occupancy_logits"].shape == (2, 30, 15, 27)


def test_kv_memory_retrieval_and_integration():
    kv = KvMemoryLexicon(channels=8, compiled=_lexicon())
    feats = torch.randn(2, 15, 8, requires_grad=True)
    out = kv(feats, torch.zeros(2, 15, N_LETTERS))
    assert out.tokens.shape == (2, 2, 8)
    assert out.cell_residual is None  # a lane-level hint, no per-cell residual

    # The per-word value memory is frozen; addressing + readout are trainable.
    assert "value_mem" in dict(kv.named_buffers())
    assert "value_mem" not in dict(kv.named_parameters())
    out.tokens.sum().backward()
    assert kv.query.weight.grad.abs().sum() > 0
    assert kv.subkeys1.grad.abs().sum() > 0
    assert kv.readout.weight.grad.abs().sum() > 0

    out = _model_with(KvMemoryLexicon(channels=8, compiled=_lexicon()))(
        torch.zeros(2, 31, 15, 15), torch.rand(2, 27)
    )
    assert out["lane_occupancy_logits"].shape == (2, 30, 15, 27)


def test_parse_module_opts():
    assert parse_module_opts(["topk=32", "x=1.5", "name=foo"]) == {
        "topk": 32,
        "x": 1.5,
        "name": "foo",
    }
    with pytest.raises(ValueError):
        parse_module_opts(["badopt"])


def test_output_contract_and_frozen_buffers():
    c = 16
    mod = SoftTraversalLexicon(channels=c, compiled=_lexicon(), topk=8)
    feats = torch.randn(4, 15, c, requires_grad=True)
    out = mod(feats, torch.zeros(4, 15, N_LETTERS))

    assert out.cell_residual.shape == (4, 15, c)
    assert out.tokens.shape == (4, mod.n_tokens, c)
    assert out.cell_signals.shape == (4, 15, 1 + N_LETTERS + 1)

    # The compiled lexicon lives in buffers (frozen), not parameters.
    param_names = {n for n, _ in mod.named_parameters()}
    buffer_names = {n for n, _ in mod.named_buffers()}
    assert {"next_tbl", "accept_tbl", "exists_tbl"} <= buffer_names
    assert not any("tbl" in n for n in param_names)

    # Gradients reach the lane features and the trainable adapters.
    out.cell_residual.sum().backward()
    assert feats.grad.abs().sum() > 0
    assert mod.query.weight.grad.abs().sum() > 0


def _force_query(mod, letter):
    """Pin the soft query to ~one-hot on `letter` and disable spurious restarts."""
    with torch.no_grad():
        mod.query.weight.zero_()
        mod.query.bias.zero_()
        mod.query.bias[ord(letter) - ord("A")] = 20.0
        mod.restart.weight.zero_()
        mod.restart.bias.fill_(-20.0)


def test_acceptance_semantics():
    mod = SoftTraversalLexicon(channels=16, compiled=_lexicon(), topk=8)
    feats = torch.zeros(1, 15, 16)

    # Board "C_T": acceptance at the final T fires only when the gap spells a word.
    _force_query(mod, "A")  # CAT is a word
    assert mod(feats, _board("CAT", gaps=(1,))).cell_signals[0, 2, 0] > 0.9
    for non in ("B", "R"):  # CBT / CRT are not words
        _force_query(mod, non)
        assert mod(feats, _board("CAT", gaps=(1,))).cell_signals[0, 2, 0] < 0.1

    # Fully on-board CARE: acceptance peaks at CAR (cell 2) and CARE (cell 3).
    _force_query(mod, "A")
    accept = mod(feats, _board("CARE")).cell_signals[0, :, 0]
    assert accept[2] > 0.9 and accept[3] > 0.9
    assert accept[0] < 0.1 and accept[1] < 0.1 and accept[4] < 0.1


def test_model_integration_matches_shapes():
    sp, sc, b = 31, 27, 2
    spatial = torch.zeros(b, sp, 15, 15)
    scalar = torch.rand(b, sc)
    targets = {
        "lane_occupancy": (torch.rand(b, 30, 15, 27) > 0.8).float(),
        "lane_score": torch.randint(0, 100, (b, 30)),
        "lane_mask": (torch.rand(b, 30) > 0.3).float(),
    }

    base = MaxMovePerLaneModel(sp, sc, trunk_channels=32, num_blocks=2, lane_layers=2)
    mod = SoftTraversalLexicon(channels=32, compiled=_lexicon(), topk=8)
    # No FFN override (full width), so the module is purely additive; the shrink
    # is covered separately by test_lane_ffn_mult_override.
    withlex = MaxMovePerLaneModel(
        sp, sc, trunk_channels=32, num_blocks=2, lane_layers=2, lexicon_module=mod
    )

    out_base = base(spatial, scalar)
    out_lex = withlex(spatial, scalar)
    assert {k: v.shape for k, v in out_base.items()} == {k: v.shape for k, v in out_lex.items()}

    # In add mode the module adds parameters and trains end-to-end.
    assert sum(p.numel() for p in withlex.parameters()) > sum(p.numel() for p in base.parameters())
    compute_loss(out_lex, targets, lambda_occ=100.0)["total"].backward()
    assert withlex.lexicon_module.query.weight.grad.abs().sum() > 0


def _lane_ffn_width(model):
    """The lane transformer's FFN hidden width (where an internal lexicon lives)."""
    return model.lane.encoder.layers[0].linear1.out_features


def test_resolve_lane_ffn_mult():
    # "add" mode never overrides the FFN width.
    assert (
        resolve_lane_ffn_mult("add", has_module=True, starve_ffn=False, replace_ffn_mult=1) is None
    )
    assert (
        resolve_lane_ffn_mult("add", has_module=False, starve_ffn=True, replace_ffn_mult=1) is None
    )

    # "replace" + tool -> shrink to replace_ffn_mult.
    assert (
        resolve_lane_ffn_mult("replace", has_module=True, starve_ffn=False, replace_ffn_mult=1) == 1
    )
    assert (
        resolve_lane_ffn_mult("replace", has_module=True, starve_ffn=False, replace_ffn_mult=2) == 2
    )

    # "replace" + no tool -> override only with the starve flag (the control).
    assert (
        resolve_lane_ffn_mult("replace", has_module=False, starve_ffn=False, replace_ffn_mult=1)
        is None
    )
    assert (
        resolve_lane_ffn_mult("replace", has_module=False, starve_ffn=True, replace_ffn_mult=1) == 1
    )


def test_lane_ffn_mult_override():
    sp, sc, c = 31, 27, 32

    def make(lane_ffn_mult):
        return MaxMovePerLaneModel(
            sp,
            sc,
            trunk_channels=c,
            num_blocks=1,
            lane_layers=1,
            ffn_mult=4,
            lane_ffn_mult=lane_ffn_mult,
        )

    assert _lane_ffn_width(make(None)) == 4 * c  # no override -> default ffn_mult width
    assert _lane_ffn_width(make(1)) == c  # shrunk

    # An override of 0 is attention-only, clamped to a 1-unit FFN, and still runs.
    m0 = make(0)
    assert _lane_ffn_width(m0) == 1
    out = m0(torch.zeros(2, sp, 15, 15), torch.rand(2, sc))
    assert out["lane_occupancy_logits"].shape == (2, 30, 15, 27)
