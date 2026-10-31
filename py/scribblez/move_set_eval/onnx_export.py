"""Export a trained MoveSetEvalModel to the plain (evidence-free) ONNX graph.

The graph is specialized to one position (P=1): its board inputs plus M
candidate-move rows, with M the only dynamic axis ("moves"). The agent scores
one position at a time, and at P=1 the training forward's padded (P, maxK, C)
query grid collapses: the scatter/gather become a reshape, `move_pos_id`
disappears, and there is no data-dependent maxK to sync on.

Move inputs keep move_set_encoder.h's native dtypes (int32 letters/squares,
uint8 masks) so the engine can feed its buffers without conversion; the casts
the torch modules need are traced into the graph.

TensorRT-refit rules, shared by every exporter here: export_graph traces with
`dynamo=False` and `do_constant_folding=False`, which keep every weight a
plain named initializer the TensorRT refitter can map. Two modules are also
rebuilt over the trained
weights as the wrapper's own parameters, because their traced forms defeat
the refitter or the tracer:

  * nn.MultiheadAttention's packed in_proj traces as a bare Constant the
    refitter cannot map, so it becomes plain q/k/v Linears plus explicit
    attention math (split_mha_qkv, cross_attention_2d).
  * The head's Linear over cat([attended, g]) is split into two Linears whose
    outputs are summed (split_concat_linear). The per-move `g` is then never
    Expanded across the dynamic M axis, which the TorchScript tracer would
    otherwise bake into a fixed shape.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.dataset import row_layout
from scribblez.onnx_export_util import (
    common_metadata,
    cross_attention_2d,
    export_graph,
    split_concat_linear,
    split_mha_qkv,
)
from scribblez.spatial_trunk import mean_max_pool

from .dataset import adopt_information_condition
from .model import MoveSetEvalModel
from .moves import move_cross_slots, move_encoding_dims
from .targets import MSET_FLAG_OPEN_LEAVES, read_mset_flags

MOVE_INPUT_NAMES = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_cross_cells",
    "move_cross_letters",
)
OUTPUT_NAMES = ("wld", "score_diff")


def board_and_summary(
    board_pos_emb: torch.Tensor, x: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trunk's outputs at P=1, x (1, C, 15, 15) and s (1, C), as the board
    token map (1, 225, C) and the global summary (1, 3C)."""
    board = x.flatten(2).transpose(1, 2) + board_pos_emb
    return board, torch.cat([mean_max_pool(x), s], dim=1)


def encode_moves_at_position(
    move_encoder: nn.Module,
    cross_check_encoder: nn.Module,
    board0: torch.Tensor,
    move_letters: torch.Tensor,
    move_blanks: torch.Tensor,
    move_squares: torch.Tensor,
    move_tile_mask: torch.Tensor,
    move_scalars: torch.Tensor,
    move_cross_cells: torch.Tensor,
    move_cross_letters: torch.Tensor,
) -> torch.Tensor:
    """model.encode_moves_on_board for M moves of the one position whose board
    tokens are board0 (225, C), from the engine's native dtypes -> (M, C):
    the single-position exports' form, with no pos_id gather."""
    tile_board = board0[move_squares.long()]  # (M, T, C)
    tile_board = tile_board * move_scalars[:, 2].view(-1, 1, 1)  # is_play gate
    e = move_encoder(
        move_letters.long(), move_blanks, move_tile_mask.float(), move_scalars, tile_board
    )
    return e + cross_check_encoder.on_position(move_cross_cells, move_cross_letters, board0)


class MoveSetEvalExportModel(nn.Module):
    """MoveSetEvalModel.forward at P=1, without evidence, sharing
    the trained model's submodules. Returns (wld (M, 3) logits, score_diff
    (M, 2) [mean, std > 0]) as a tuple to fix the exported output order.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        self.trunk = model.trunk
        self.board_pos_emb = model.board_pos_emb
        self.move_encoder = model.move_encoder
        self.cross_check_encoder = model.cross_check_encoder
        # Rebuilt for the refitter (module docstring); out_proj is already plain.
        mha = model.cross_attn
        c = mha.embed_dim
        self.num_heads = mha.num_heads
        self.q_proj, self.k_proj, self.v_proj = split_mha_qkv(mha)
        self.attn_out = mha.out_proj
        # head[0] is Linear(cat([attended (C), g (3C)])); split at C.
        self.head_attended, self.head_g = split_concat_linear(model.head[0], c)
        self.head_out = model.head[2]

    def _cross_attention(self, e: torch.Tensor, board0: torch.Tensor) -> torch.Tensor:
        """e (M, C) move queries over board0 (225, C) -> (M, C)."""
        return cross_attention_2d(
            self.q_proj, self.k_proj, self.v_proj, self.attn_out, self.num_heads, e, board0
        )

    def forward(
        self,
        input_spatial: torch.Tensor,  # (1, planes, 15, 15) f32
        input_scalar: torch.Tensor,  # (1, S) f32
        move_letters: torch.Tensor,  # (M, T) i32
        move_blanks: torch.Tensor,  # (M, T) u8
        move_squares: torch.Tensor,  # (M, T) i32
        move_tile_mask: torch.Tensor,  # (M, T) u8
        move_scalars: torch.Tensor,  # (M, 3) f32
        move_cross_cells: torch.Tensor,  # (M, K) i32
        move_cross_letters: torch.Tensor,  # (M, K * 26) u8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, s = self.trunk(input_spatial, input_scalar)  # (1,C,15,15), (1,C)
        board, g = board_and_summary(self.board_pos_emb, x, s)
        e = encode_moves_at_position(
            self.move_encoder, self.cross_check_encoder, board[0], move_letters, move_blanks,
            move_squares, move_tile_mask, move_scalars, move_cross_cells, move_cross_letters,
        )  # fmt: skip

        attended = self._cross_attention(e, board[0])
        h = F.relu(self.head_attended(attended) + self.head_g(g))
        out = self.head_out(h)  # (M, 5)
        sd_mean = out[:, 3:4]
        sd_std = F.softplus(out[:, 4:5]) + 1e-3
        return out[:, :3], torch.cat([sd_mean, sd_std], dim=1)


def export_onnx(
    model: MoveSetEvalModel,
    path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    *,
    opp_leave_input: bool,
    move_encoding_version: int,
    board_size: int = 15,
    opset: int = 17,
):
    """Write the plain graph to `path` atomically. The metadata records the
    input-encoding arm, `graph=move_set_eval` and the move-encoding version;
    the engine checks the version so a model never runs against move rows it
    was not trained on."""
    device = next(model.parameters()).device
    # Moves the wrapper's rebuilt Linears, which are constructed on the CPU.
    wrapper = MoveSetEvalExportModel(model).to(device)
    t, _, _, _ = move_encoding_dims()
    k = move_cross_slots()

    dummy_m = 5  # any M > 1; the parity tests assert other Ms against it
    dummies = (
        torch.zeros(1, spatial_planes, board_size, board_size, device=device),
        torch.zeros(1, scalar_size, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, 3, device=device),
        torch.zeros(dummy_m, k, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, k * 26, dtype=torch.uint8, device=device),
    )
    export_graph(
        wrapper,
        dummies,
        path,
        input_names=["input_spatial", "input_scalar", *MOVE_INPUT_NAMES],
        output_names=OUTPUT_NAMES,
        dynamic_axes={name: {0: "moves"} for name in (*MOVE_INPUT_NAMES, *OUTPUT_NAMES)},
        metadata={
            **common_metadata(opp_leave_input),
            "graph": "move_set_eval",
            "move_encoding_version": str(move_encoding_version),
        },
        opset=opset,
    )


def legacy_checkpoint_condition(paths) -> dict:
    """Reconstruct the config fields a checkpoint lacks when it was saved
    without them: the information condition from the tag's .mset corpus, the
    input widths from the FFI session layout, and move_encoding_version 0
    (which a version-checking engine loader refuses against a newer encoder).
    Raises FileNotFoundError when the tag has no corpus."""
    mset_files = sorted(Path(paths.data_dir).glob("slogs/*.mset"))
    if not mset_files:
        raise FileNotFoundError(
            f"checkpoint config predates the self-describing fields and "
            f"{Path(paths.data_dir) / 'slogs'} holds no .mset to re-adopt the arm from"
        )
    adopt_information_condition(mset_files)
    input_shapes, _ = row_layout()
    dims = {s.name: s.dims for s in input_shapes}
    return {
        "open_leaves": bool(read_mset_flags(mset_files[0]) & MSET_FLAG_OPEN_LEAVES),
        "spatial_planes": dims["input_spatial"][0],
        "scalar_size": dims["input_scalar"][0],
        "move_encoding_version": 0,
    }
