"""The transformer trunk tower (transformer_tower.py) in PositionEvalModel, with
its tile-supply register tokens (supply_registers.py).

Pins the contracts the tower is built on: every block starts as the identity on
the trunk stream (zero-initialised up-projections), the register tokens are built
from the right scalar slices under both encoding arms, gradients reach the
registers' content and positions and the rotary frequencies, and the graph
exports through the pinned legacy ONNX exporter and matches torch at a batch
size other than the traced one.
"""

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES, PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.position_eval.supply_registers import (
    N_TILES,
    OPP_LEAVE0,
    SCALAR_SIZE_HIDDEN_LEAVES,
    SCALAR_SIZE_OPEN_LEAVES,
    TILE_COUNTS,
    UNSEEN_THERMO0,
    TileSupplyRegisters,
    _thermometer_to_count_matrix,
)
from scribblez.transformer_tower import TransformerConfig

P = 87
C = 32
CFG = TransformerConfig(mid_channels=16, num_heads=4, ffn_channels=32)


def _model(scalar_size=SCALAR_SIZE_OPEN_LEAVES, num_blocks=2):
    torch.manual_seed(0)
    return PositionEvalModel(
        P, scalar_size, trunk_channels=C, num_blocks=num_blocks, transformer=CFG
    ).eval()


def _inputs(batch, scalar_size=SCALAR_SIZE_OPEN_LEAVES, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, P, 15, 15, generator=g),
        torch.randn(batch, scalar_size, generator=g),
    )


@pytest.mark.parametrize("scalar_size", [SCALAR_SIZE_HIDDEN_LEAVES, SCALAR_SIZE_OPEN_LEAVES])
def test_forward_shapes_under_both_arms(scalar_size):
    """Both encoding arms build and run: the registers carry the opp-leave seat
    only under the open-leaves arm."""
    model = _model(scalar_size)
    assert model.trunk.registers.has_opp_leave == (scalar_size == SCALAR_SIZE_OPEN_LEAVES)
    out = model(*_inputs(3, scalar_size))
    assert out["wld"].shape == (3, 3)
    for name in PLACEMENT_HEAD_NAMES:
        assert out[name].shape[0] == 3


def test_conv_trunk_has_no_registers():
    model = PositionEvalModel(P, SCALAR_SIZE_OPEN_LEAVES, trunk_channels=C, num_blocks=2)
    assert model.trunk.tower is None and model.trunk.registers is None


def test_blocks_start_as_identity():
    """Every block's up-projection is zero-initialised, so at init the tower is
    the final norm over the stem features: each block's residual is exactly 0."""
    model = _model()
    tower = model.trunk.tower
    tokens = torch.randn(2, 225 + N_TILES, C)
    pos_x, pos_y = tower._positions()
    for block in tower.blocks:
        assert torch.count_nonzero(block(tokens, pos_x, pos_y)) == 0
    # ... and stays live once the projection moves off zero.
    with torch.no_grad():
        tower.blocks[0].up.weight.normal_(std=0.1)
    assert torch.count_nonzero(tower.blocks[0](tokens, pos_x, pos_y)) > 0


def test_gradients_reach_registers_and_rope():
    """A placement loss trains the register content, the register positions, and
    the rotary frequencies -- the parts a wiring slip would silently detach."""
    model = _model()
    with torch.no_grad():
        for block in model.trunk.tower.blocks:
            block.up.weight.normal_(std=0.1)
    out = model(*_inputs(2))
    F.cross_entropy(out["opp_next_placement"], torch.zeros(2, dtype=torch.long)).backward()
    params = dict(model.named_parameters())
    for name in (
        "trunk.registers.token_embed",
        "trunk.registers.supply_proj.weight",
        "trunk.tower.register_pos",
        "trunk.tower.blocks.0.pairs.0.attention.rope_freqs",
        "trunk.tower.blocks.1.pairs.1.attention.q_proj.weight",
        "trunk.tower.blocks.1.pairs.1.ffn.gate.weight",
    ):
        g = params[name].grad
        assert g is not None and torch.count_nonzero(g) > 0, f"no gradient to {name}"


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(mid_channels=16, num_heads=3, ffn_channels=32)
    with pytest.raises(ValueError, match="even"):
        TransformerConfig(mid_channels=12, num_heads=4, ffn_channels=32)
    with pytest.raises(ValueError, match="scalar_size"):
        TileSupplyRegisters(C, 100)


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


def test_registers_read_the_right_scalar_slices():
    """A single letter's availability, injected into each of the three scalar
    seats in turn, moves only that letter's token, and differently per seat --
    proving rack, unseen, and opp-leave slices are all wired in and kept distinct."""
    torch.manual_seed(0)
    regs = TileSupplyRegisters(C, SCALAR_SIZE_OPEN_LEAVES)
    li = 18  # 'S'
    with torch.no_grad():
        base = regs(torch.zeros(1, SCALAR_SIZE_OPEN_LEAVES))[0]
        contents = []
        for slot in (li, UNSEEN_THERMO0 + sum(TILE_COUNTS[:li]), OPP_LEAVE0 + li):
            sc = torch.zeros(1, SCALAR_SIZE_OPEN_LEAVES)
            sc[0, slot] = 1.0
            tokens = regs(sc)[0]
            moved = (tokens != base).any(dim=-1)
            assert moved[li] and moved.sum() == 1, f"slot {slot} moved tokens {moved.nonzero()}"
            contents.append(tokens[li])
    assert not torch.allclose(contents[0], contents[1])
    assert not torch.allclose(contents[1], contents[2])
    assert not torch.allclose(contents[0], contents[2])


def test_onnx_export_matches_torch(tmp_path):
    """The tower exports through the pinned legacy exporter (manual attention,
    elementwise RMSNorm, in-graph RoPE tables) and matches torch at a batch size
    other than the traced one."""
    model = _model()
    with torch.no_grad():
        for block in model.trunk.tower.blocks:
            block.up.weight.normal_(std=0.1)
    sp, sc = _inputs(3)
    path = tmp_path / "m.onnx"
    export_onnx(model, path, P, SCALAR_SIZE_OPEN_LEAVES, opp_leave_input=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {"input_spatial": sp.numpy(), "input_scalar": sc.numpy()}
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed), strict=True))
    with torch.no_grad():
        ref = model(sp, sc)
    for name in ["wld", "score_diff", *PLACEMENT_HEAD_NAMES]:
        assert np.abs(outs[name] - ref[name].numpy()).max() < 1e-4, name
