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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.dataset import row_layout
from scribblez.ffi import score_diff_input_layout
from scribblez.fp16_gate import PROBE_LEADS, check_fp16_headroom
from scribblez.onnx_export_util import (
    architecture_signature,
    atomic_output,
    common_metadata,
    cross_attention_2d,
    split_concat_linear,
    split_mha_qkv,
    undo_initializer_dedup,
    write_metadata,
)
from scribblez.spatial_trunk import mean_max_pool

from .dataset import adopt_information_condition
from .model import MoveSetEvalModel
from .moves import move_encoding_dims
from .targets import MSET_FLAG_OPEN_LEAVES, read_mset_flags

MOVE_INPUT_NAMES = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
)
# The P=1 export deliberately omits the placement-plane readout: the engine
# has no consumer for per-candidate planes until the evidence path lands
# (roadmap item 3), which also decides how the cached-vs-per-iteration graph
# split exposes them. Until then planes are a training-time distillation head.
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
        # The cross-attention is re-expressed from the trained
        # nn.MultiheadAttention as plain q/k/v Linears (split_mha_qkv): tracing
        # the packed in_proj leaves a bare Constant with no initializer behind
        # it, which the TensorRT parser-refitter cannot refit (the refit gate
        # probe caught exactly this); three plain Linears export as plain named
        # initializers. out_proj is already a plain Linear.
        mha = model.cross_attn
        c = mha.embed_dim
        self.num_heads = mha.num_heads
        self.q_proj, self.k_proj, self.v_proj = split_mha_qkv(mha)
        self.attn_out = mha.out_proj
        # Re-associate head[0] = Linear(cat([attended (C), g (3C)])) into an
        # attended sub-Linear (keeps the bias) and a bias-free g sub-Linear, so
        # both export as plain initializers and the per-move g never Expands.
        self.head_attended, self.head_g = split_concat_linear(model.head[0], c)
        self.head_out = model.head[2]

    def _cross_attention(self, e: torch.Tensor, board0: torch.Tensor) -> torch.Tensor:
        """nn.MultiheadAttention's math (batch_first, no mask, eval mode) over
        M move queries (e, (M, C)) and one position's 225 board tokens
        (board0, (225, C)) -> (M, C)."""
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
    opp_leave_input: bool,
    move_encoding_version: int,
    board_size: int = 15,
    opset: int = 17,
    probe_feeds: list[dict] | None = None,
) -> float | None:
    """Wrap `model` in the P=1 export forward, trace it with a dynamic "moves"
    axis, and write the ONNX graph to `path` atomically, stamping the
    input-encoding arm, `graph=move_set_eval`, and the move-encoding version
    into its metadata_props (the version is what stops a checkpoint from
    silently running against an encoder whose rows it was not trained on).

    probe_feeds, when given (see fp16_probe_feeds_from_batch), runs the
    FP16-headroom gate on the written graph: an overflowing checkpoint raises
    Fp16HeadroomError and no file lands at `path`. Returns the probe's peak
    |activation| (the trainer's per-export growth series), or None when not
    probed."""
    path = Path(path)
    was_training = model.training
    model.eval()  # explicit, not via the wrapper's aliasing of the submodules
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
                **common_metadata(opp_leave_input),
                "model-architecture-signature": architecture_signature(wrapper, opset),
                "graph": "move_set_eval",
                "move_encoding_version": str(move_encoding_version),
            },
        )
        peak = check_fp16_headroom(tmp_path, probe_feeds) if probe_feeds is not None else None
    if was_training:
        model.train()
    return peak


# Positions a probe batch contributes to the gate feeds. Each yields
# 1 + len(PROBE_LEADS) P=1 forwards, so the probe stays a few dozen runs.
FP16_PROBE_POSITIONS = 8


def fp16_probe_feeds_from_batch(batch: dict, *, leads: tuple[int, ...] = PROBE_LEADS) -> list[dict]:
    """Gate probe feeds from a training batch dict (MsetDataset / the evidence
    dataset -- any dict with the board and move input keys): the first
    FP16_PROBE_POSITIONS positions, each as one P=1 feed per lead in `leads`
    (the board's score-diff scalar stamped, exactly the engine's own
    encode_score_diff_sweep transform) plus one as encoded. The move rows keep
    their encoded scalars: the measured overflow site is the board trunk's
    pooled branch, which the board inputs alone drive."""
    sd_index, sd_scale = score_diff_input_layout()
    spatial = batch["input_spatial"].numpy()
    scalar = batch["input_scalar"].numpy()
    pos_id = batch["move_pos_id"].numpy()
    moves = {
        "move_letters": batch["move_letters"].numpy().astype(np.int32),
        "move_blanks": batch["move_blanks"].numpy().astype(np.uint8),
        "move_squares": batch["move_squares"].numpy().astype(np.int32),
        "move_tile_mask": batch["move_tile_mask"].numpy().astype(np.uint8),
        "move_scalars": batch["move_scalars"].numpy().astype(np.float32),
    }
    feeds = []
    for p in range(min(FP16_PROBE_POSITIONS, spatial.shape[0])):
        rows = pos_id == p
        move_feed = {name: np.ascontiguousarray(arr[rows]) for name, arr in moves.items()}
        scalars = [scalar[p]]
        for lead in leads:
            stamped = scalar[p].copy()
            stamped[sd_index] = lead / sd_scale
            scalars.append(stamped)
        for sc in scalars:
            feeds.append(
                {
                    "input_spatial": np.ascontiguousarray(spatial[p : p + 1]),
                    "input_scalar": sc[None],
                    **move_feed,
                }
            )
    return feeds


def legacy_checkpoint_condition(paths) -> dict:
    """Recover the self-describing config fields for a checkpoint that predates
    them: adopt the information-condition arm from the tag's .mset corpus (the
    trainer's own path), then read the input widths off the session layout.
    Raises FileNotFoundError when the tag holds no corpus to read the arm from.
    The recovered move_encoding_version is 0 -- pre-exchange-fix rows -- which
    an engine loader enforcing the version will rightly refuse to run against
    a newer encoder."""
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
