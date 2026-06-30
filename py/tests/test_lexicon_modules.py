"""Tests for the compiled-lexicon modules: the registry, the soft-traversal
module's contract and semantics, and its integration into the lane model."""

import pytest
import torch
from scribblez.max_move_per_lane.lexicon_compiler import N_LETTERS, CompiledLexicon
from scribblez.max_move_per_lane.lexicon_modules import (
    SoftTraversalLexicon,
    available_modules,
    build_lexicon_module,
    parse_module_opts,
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
    assert "soft_traversal" in available_modules()
    assert build_lexicon_module("none", channels=8, kwg_path="/nonexistent") is None
    with pytest.raises(KeyError):
        build_lexicon_module("bogus", channels=8, kwg_path="/nonexistent")


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
    withlex = MaxMovePerLaneModel(
        sp, sc, trunk_channels=32, num_blocks=2, lane_layers=2, lexicon_module=mod
    )

    out_base = base(spatial, scalar)
    out_lex = withlex(spatial, scalar)
    assert {k: v.shape for k, v in out_base.items()} == {k: v.shape for k, v in out_lex.items()}

    # The module adds parameters and trains end-to-end.
    assert sum(p.numel() for p in withlex.parameters()) > sum(p.numel() for p in base.parameters())
    compute_loss(out_lex, targets, lambda_occ=100.0)["total"].backward()
    assert withlex.lexicon_module.query.weight.grad.abs().sum() > 0
