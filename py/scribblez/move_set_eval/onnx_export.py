"""Export a trained MoveSetEvalModel to ONNX (docs/roadmap.md, A4).

The exported graph is the P=1 specialization: one position's board inputs plus
M candidate-move rows, with M the single dynamic axis ("moves"). At a decision
point the agent holds exactly one position, and under P=1 the training
forward's padded (P, maxK, C) query grid degenerates exactly -- the
scatter/gather become a reshape, `move_pos_id` disappears, and the
data-dependent maxK host sync never arises -- so the export carries none of
the batching machinery, only the model.

Move-input dtypes match move_set_encoder.h's native buffers (int32
letters/squares, uint8 masks) so the engine feeds them zero-copy; the
int64/float casts the torch modules need are traced into the graph.

Weight-map discipline (shared with the position exporter): `dynamo=False` and
`do_constant_folding=False` keep every weight a plain named initializer for
the TensorRT parser-refitter. Two modules are additionally rebuilt over the
trained weights as the wrapper's OWN parameters, because their traced forms
defeat the refitter: the cross-attention (nn.MultiheadAttention's packed
in_proj surfaces in the trace as a bare Constant the refitter cannot map --
caught by the refit gate probe) becomes three plain q/k/v Linears plus
explicit attention math, and the head's concat-Linear is re-associated into
attended + g sub-Linears, which also removes the per-move `g` Expand, the
legacy tracer's main shape-baking hazard.
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.ffi import DEFAULT_LEXICON
from scribblez.onnx_export_util import (
    architecture_signature,
    atomic_output,
    undo_initializer_dedup,
    write_metadata,
)
from scribblez.spatial_trunk import mean_max_pool

from .model import MoveSetEvalModel
from .moves import move_encoding_dims

MOVE_INPUT_NAMES = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
)
OUTPUT_NAMES = ("wld", "score_diff")


class MoveSetEvalExportModel(nn.Module):
    """The P=1 forward over a trained model's own submodules.

    Board inputs are (1, ...) and the M move rows query the single position's
    board tokens directly -- `MoveSetEvalModel.forward`'s grid construction
    applied at P=1, where `rank == arange(M)` and the scatter is an unsqueeze.
    Outputs are the training heads' (M, 3) WLD logits and (M, 2) score-diff
    [mean, std>0], as a tuple so the exported output order is explicit.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        self.trunk = model.trunk
        self.board_pos_emb = model.board_pos_emb
        self.move_encoder = model.move_encoder
        # The cross-attention is hand-rolled from the trained
        # nn.MultiheadAttention's own weights, with q/k/v as this wrapper's own
        # nn.Linear parameters (copied slices of the packed in_proj). Tracing
        # nn.MultiheadAttention leaves an in_proj-derived tensor in the graph
        # as a bare Constant with no initializer behind it, which the TensorRT
        # parser-refitter cannot refit (the refit gate probe caught exactly
        # this); three plain Linears export as plain named initializers.
        mha = model.cross_attn
        c = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = c // mha.num_heads
        self.q_proj = nn.Linear(c, c)
        self.k_proj = nn.Linear(c, c)
        self.v_proj = nn.Linear(c, c)
        with torch.no_grad():
            self.q_proj.weight.copy_(mha.in_proj_weight[:c])
            self.q_proj.bias.copy_(mha.in_proj_bias[:c])
            self.k_proj.weight.copy_(mha.in_proj_weight[c : 2 * c])
            self.k_proj.bias.copy_(mha.in_proj_bias[c : 2 * c])
            self.v_proj.weight.copy_(mha.in_proj_weight[2 * c :])
            self.v_proj.bias.copy_(mha.in_proj_bias[2 * c :])
        self.attn_out = mha.out_proj  # a plain Linear already
        # Re-associate head[0] = Linear(cat([attended (C), g (3C)])): the
        # attended block keeps the bias, the g block is bias-free, and both are
        # this wrapper's own named parameters (copied slices) so they export as
        # plain initializers.
        first = model.head[0]
        self.head_attended = nn.Linear(c, first.out_features)
        self.head_g = nn.Linear(first.in_features - c, first.out_features, bias=False)
        with torch.no_grad():
            self.head_attended.weight.copy_(first.weight[:, :c])
            self.head_attended.bias.copy_(first.bias)
            self.head_g.weight.copy_(first.weight[:, c:])
        self.head_out = model.head[2]

    def _cross_attention(self, e: torch.Tensor, board0: torch.Tensor) -> torch.Tensor:
        """nn.MultiheadAttention's math (batch_first, no mask, eval mode) over
        M move queries (e, (M, C)) and one position's 225 board tokens
        (board0, (225, C)) -> (M, C)."""
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(e).view(-1, h, d).transpose(0, 1)  # (H, M, d)
        k = self.k_proj(board0).view(-1, h, d).transpose(0, 1)  # (H, 225, d)
        v = self.v_proj(board0).view(-1, h, d).transpose(0, 1)
        attn = torch.softmax(q @ k.transpose(1, 2) * d**-0.5, dim=-1)  # (H, M, 225)
        ctx = (attn @ v).transpose(0, 1).reshape(-1, h * d)  # (M, C)
        return self.attn_out(ctx)

    def forward(
        self,
        input_spatial: torch.Tensor,  # (1, planes, 15, 15) f32
        input_scalar: torch.Tensor,  # (1, S) f32
        move_letters: torch.Tensor,  # (M, T) i32
        move_blanks: torch.Tensor,  # (M, T) u8
        move_squares: torch.Tensor,  # (M, T) i32
        move_tile_mask: torch.Tensor,  # (M, T) u8
        move_scalars: torch.Tensor,  # (M, 3) f32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        letters = move_letters.long()
        squares = move_squares.long()
        tile_mask = move_tile_mask.float()

        x, s = self.trunk(input_spatial, input_scalar)  # (1,C,15,15), (1,C)
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (1, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)  # (1, 3C)

        # The single position's board token at each tile's square, gated by
        # is_play exactly as the training forward gates it (exchange tiles
        # carry letters but no squares).
        tile_board = board[0][squares]  # (M, T, C)
        tile_board = tile_board * move_scalars[:, 2].view(-1, 1, 1)
        e = self.move_encoder(letters, move_blanks, tile_mask, move_scalars, tile_board)

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
    contingent_features: bool,
    opp_leave_input: bool,
    move_encoding_version: int,
    board_size: int = 15,
    opset: int = 17,
):
    """Wrap `model` in the P=1 export forward, trace it with a dynamic "moves"
    axis, and write the ONNX graph to `path` atomically, stamping the
    input-encoding arm, `graph=move_set_eval`, and the move-encoding version
    into its metadata_props (the version is what stops a checkpoint from
    silently running against an encoder whose rows it was not trained on)."""
    path = Path(path)
    was_training = model.training
    device = next(model.parameters()).device
    # .to(device) is a no-op for the shared trained modules and moves only the
    # wrapper's own materialized sub-Linears, which construct on the CPU.
    wrapper = MoveSetEvalExportModel(model).to(device)
    wrapper.eval()
    t, _, _, _ = move_encoding_dims()

    dummy_m = 5  # any M > 1; the parity tests assert other Ms against it
    dummies = (
        torch.zeros(1, spatial_planes, board_size, board_size, device=device),
        torch.zeros(1, scalar_size, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, t, dtype=torch.int32, device=device),
        torch.zeros(dummy_m, t, dtype=torch.uint8, device=device),
        torch.zeros(dummy_m, 3, device=device),
    )
    input_names = ["input_spatial", "input_scalar", *MOVE_INPUT_NAMES]

    # dynamo=False for the same reasons as the position exporter (see its
    # export_onnx); DeprecationWarnings silenced likewise.
    with atomic_output(path) as tmp_path, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            wrapper,
            dummies,
            str(tmp_path),
            input_names=input_names,
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={name: {0: "moves"} for name in (*MOVE_INPUT_NAMES, *OUTPUT_NAMES)},
            opset_version=opset,
            dynamo=False,
            do_constant_folding=False,
        )
        undo_initializer_dedup(tmp_path)
        write_metadata(
            tmp_path,
            {
                "contingent_features": "true" if contingent_features else "false",
                "opp_leave_input": "true" if opp_leave_input else "false",
                "lexicon": DEFAULT_LEXICON,
                "model-architecture-signature": architecture_signature(wrapper, opset),
                "graph": "move_set_eval",
                "move_encoding_version": str(move_encoding_version),
            },
        )
    if was_training:
        model.train()
