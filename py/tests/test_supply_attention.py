"""Tile-supply cross-attention in PositionEvalModel.

use_supply_attention inserts one cross-attention block on the post-trunk feature
map: board squares attend to per-letter availability tokens so the placement
heads can gate cross-checks on tile availability. The block's output projection
is zero-initialised, so it is inert at init -- a strict superset of the
no-attention model. These tests pin that contract (inert at init, live once
perturbed, gradients reach the supply tokens) and that the block builds its
supply tokens from the correct scalar slices.
"""

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES, PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.position_eval.supply_attention import (
    OPP_LEAVE0,
    TILE_COUNTS,
    UNSEEN_THERMO0,
    _thermometer_to_count_matrix,
)

P, S = 85, 163


def _model(use_attn):
    torch.manual_seed(0)
    return PositionEvalModel(
        P, S, trunk_channels=64, num_blocks=6, use_supply_attention=use_attn
    ).eval()


def test_block_is_inert_at_init():
    """The zero-initialised output projection makes the block the identity, so the
    model computes the same function as itself with the block bypassed. (Compared
    within one instance: a separately-built baseline diverges only because
    constructing the block consumes RNG before the heads are initialised.)"""
    model = _model(use_attn=True)
    sp, sc = torch.randn(3, P, 15, 15), torch.randn(3, S)
    with torch.no_grad():
        withb = model(sp, sc)
        block, model.supply_attention = model.supply_attention, None
        bypass = model(sp, sc)
        model.supply_attention = block
    for k in withb:
        assert torch.allclose(withb[k], bypass[k], atol=1e-6), k


def test_output_projection_zero_initialised():
    model = _model(use_attn=True)
    assert torch.count_nonzero(model.supply_attention.out_proj.weight) == 0
    assert torch.count_nonzero(model.supply_attention.out_proj.bias) == 0


def test_perturbed_block_changes_output_and_gradients_flow():
    """Off zero the block modulates (the hidden capacity is reachable), and a
    placement loss sends gradient all the way back to the supply-token builder."""
    model = _model(use_attn=True)
    sp, sc = torch.randn(2, P, 15, 15), torch.randn(2, S)
    with torch.no_grad():
        base = model(sp, sc)["opp_next_placement"]
        model.supply_attention.out_proj.weight.normal_(std=0.1)
        moved = model(sp, sc)["opp_next_placement"]
    assert not torch.allclose(base, moved), "output projection is dead"

    model.zero_grad()
    out = model(sp, sc)
    F.binary_cross_entropy_with_logits(
        out["opp_next_placement"], torch.zeros_like(out["opp_next_placement"])
    ).backward()
    for name in ("supply_proj.weight", "token_embed", "q_proj.weight", "k_proj.weight"):
        g = dict(model.supply_attention.named_parameters())[name].grad
        assert g is not None and torch.count_nonzero(g) > 0, f"no gradient to {name}"


def test_requires_open_leaves_arm():
    """The block reads the opp-leave scalar block, so it rejects the hidden-leaves
    arm (which has no such block)."""
    import pytest

    with pytest.raises(ValueError, match="open-leaves"):
        PositionEvalModel(P, 136, use_supply_attention=True)


def test_thermometer_decode_recovers_counts():
    """The fixed thermometer->count matrix inverts the encoder's per-letter
    thermometer: a region with k hot slots decodes to count k."""
    m = _thermometer_to_count_matrix()
    thermo = torch.zeros(100)
    # Set 'S' (index 18) to 3 and 'E' (index 4) to 12 (both within their widths).
    s0 = sum(TILE_COUNTS[:18])
    e0 = sum(TILE_COUNTS[:4])
    thermo[s0 : s0 + 3] = 1.0
    thermo[e0 : e0 + 12] = 1.0
    counts = thermo @ m
    assert counts[18] == 3 and counts[4] == 12
    assert counts.sum() == 15  # nothing leaks across regions


def test_supply_tokens_read_the_right_scalar_slices():
    """A single letter's availability, injected into each of the three scalar
    seats in turn, produces three distinct supply-token contents -- proving rack,
    unseen, and opp-leave slices are all wired in and kept distinct."""
    model = _model(use_attn=True)
    block = model.supply_attention
    li = 18  # 'S'
    contents = []
    for slot in (li, UNSEEN_THERMO0 + sum(TILE_COUNTS[:li]), OPP_LEAVE0 + li):
        sc = torch.zeros(1, S)
        sc[0, slot] = 1.0
        with torch.no_grad():
            contents.append(block._supply_tokens(sc)[0, li])
    # Each seat feeds a different column of supply_proj, so the three tokens differ.
    assert not torch.allclose(contents[0], contents[1])
    assert not torch.allclose(contents[1], contents[2])
    assert not torch.allclose(contents[0], contents[2])


def test_onnx_export_matches_torch():
    """The block exports through the pinned legacy exporter (manual attention =
    matmul + softmax, no scaled_dot_product_attention) and matches torch."""
    import tempfile
    from pathlib import Path

    model = _model(use_attn=True)
    sp, sc = torch.randn(2, P, 15, 15), torch.randn(2, S)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.onnx"
        export_onnx(model, path, P, S, opp_leave_input=True)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {"input_spatial": sp.numpy(), "input_scalar": sc.numpy()}
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed), strict=True))
    with torch.no_grad():
        ref = model(sp, sc)
    for name in ["wld", "score_diff", *PLACEMENT_HEAD_NAMES]:
        assert np.abs(outs[name] - ref[name].numpy()).max() < 1e-4, name
