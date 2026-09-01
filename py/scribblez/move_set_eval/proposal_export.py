"""Export the move proposal model as the split evidence-path graphs (roadmap item 3).

The move proposal model is a MoveSetEvalModel carrying the evidence fusion stage
and the proves-best head. Its deployment loop
(docs/sim_residual_feedback.md, docs/roadmap.md) runs it incrementally: encode
the board, moves, and evidence-free predictions ONCE per turn, then -- after
each sim -- condition on the growing evidence set and re-score every candidate,
without recomputing the trunk. This module emits that split as two ONNX graphs,
mirroring MoveSetEvalModel's own staged API (encode_board / encode_moves /
score_moves vs. evidence_fusion + conditioned score_moves):

  * `move_proposal_cache` (once per turn): the P=1 board inputs plus M candidate
    rows -> the cache (board tokens, global summary, per-move encodings) and the
    evidence-free predictions (wld, score_diff, planes). M is the single dynamic
    axis.
  * `move_proposal_step` (per loop iteration): the cache tensors plus a padded
    evidence set of fixed width E -> the evidence-conditioned wld, score_diff,
    planes, and the proves-best gain. M rides the dynamic axis; the evidence
    inputs are fixed-width (leading-1 batch), so the graph keeps ONE dynamic
    axis exactly as the plain move-set graph does.

Refitter discipline (shared with onnx_export.py's MoveSetEvalExportModel, via
the helpers in onnx_export_util.py): `dynamo=False` and
`do_constant_folding=False` keep every weight a plain named initializer for the
TensorRT parser-refitter, any nn.MultiheadAttention (whose packed in_proj traces
as a bare Constant the refitter cannot map) is re-expressed into plain q/k/v
Linears plus explicit attention math (split_mha_qkv / cross_attention_2d), and a
Linear over a concatenation is re-associated (split_concat_linear) so the
per-move `g` never has to Expand across the dynamic M axis. The scoring
cross-attention (model.cross_attn) and BOTH fusion attentions get the attention
treatment; the fusion's own cross-attention, though already plain Linears, is
re-expressed here so its padding mask becomes an additive float bias (NEG_BIAS)
rather than a boolean masked_fill, keeping boolean reductions -- which the
ONNX/TensorRT path handles poorly -- out of the graph.

The evidence tokens are re-encoded inside the step graph (encode_tokens is
folded into ProposalStepExportModel rather than split into a third graph):
re-encoding the <=E tokens each iteration is negligible beside the rollouts each
step schedules, and the finer per-candidate-encode / device-resident split is
deferred to the engine runtime (see docs/sim_residual_feedback.md).
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.evidence_fusion import NUM_EVIDENCE_PLANES, NUM_EVIDENCE_SCALARS
from scribblez.footprint_spatial import CATCH_ALL, SLOTS_PER_CELL
from scribblez.onnx_export_util import (
    architecture_signature,
    atomic_output,
    common_metadata,
    cross_attention_2d,
    split_concat_linear,
    split_mha_qkv,
    undo_initializer_dedup,
    weight_fingerprint,
    write_metadata,
)
from scribblez.spatial_trunk import mean_max_pool

from .model import MoveSetEvalModel, footprint_cell_marginal
from .moves import move_encoding_dims
from .targets import PLANE_NAMES

# The padded evidence-set width the step graph is specialized to. E is baked
# into the step graph as a fixed shape (the leading-1 batch keeps M the only
# dynamic axis), so an engine loader must stage exactly this many evidence rows.
# 64 is comfortably above the deployment sim budget; a single retunable pin.
DEFAULT_MAX_EVIDENCE = 64

# ONNX `graph` metadata values -- must match the engine's kGraphMoveProposal*
# constants (engine/include/nn/onnx_metadata.h), the strings a C++ loader gates
# the runtime on.
GRAPH_CACHE = "move_proposal_cache"
GRAPH_STEP = "move_proposal_step"

# A logit bias that softmaxes to exactly zero weight -- exp(-1e9) underflows to
# 0.0 in IEEE arithmetic, so an additive `(attend - 1) * NEG_BIAS` mask is
# bit-identical to a boolean masked_fill(-inf) while staying a plain float op.
NEG_BIAS = 1.0e9

CACHE_INPUT_NAMES = (
    "input_spatial",
    "input_scalar",
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
)
CACHE_OUTPUT_NAMES = ("board", "g", "move_enc", "wld", "score_diff", "planes")

STEP_INPUT_NAMES = (
    "board",
    "g",
    "move_enc",
    "ev_move_enc",
    "ev_obs_planes",
    "ev_obs_scalars",
    "ev_mask",
)
STEP_OUTPUT_NAMES = ("wld", "score_diff", "planes", "gain")


class _ScoringHeads(nn.Module):
    """The move proposal model's scoring machinery, refitter-re-expressed.

    Scores M encoded moves against a board token map and global summary --
    plain (over the trunk map, in the cache graph) or evidence-conditioned (over
    the fused map, in the step graph); the math is identical, only the board/g
    it reads differ. `value` returns the attended embeddings alongside the WLD /
    score-diff / plane heads; `gain` reads the same attended embedding for the
    proves-best output.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        # Scoring cross-attention (model.cross_attn) as plain q/k/v/out Linears.
        mha = model.cross_attn
        c = mha.embed_dim
        self.c = c
        self.num_heads = mha.num_heads
        self.q_proj, self.k_proj, self.v_proj = split_mha_qkv(mha)
        self.attn_out = mha.out_proj  # a plain Linear already

        # Value head Linear(4C, C) over cat([attended, g]) -> attended/g split.
        self.head_attended, self.head_g = split_concat_linear(model.head[0], c)
        self.head_out = model.head[2]
        # Plane readout: the footprint head's Linear(4C, num_planes*slots*C) plane
        # queries and its Linear(4C, num_planes*catch_all) direct catch-all, each
        # attended/g split.
        self.num_planes = len(PLANE_NAMES)
        self.plane_attended, self.plane_g = split_concat_linear(model.plane_proj, c)
        self.plane_catch_attended, self.plane_catch_g = split_concat_linear(model.plane_catch, c)
        # Proves-best head Linear(4C, C) -> attended/g split, then Linear(C, 1).
        self.pb_attended, self.pb_g = split_concat_linear(model.proves_best[0], c)
        self.pb_out = model.proves_best[2]

    def _cross_attention(self, e: torch.Tensor, board0: torch.Tensor) -> torch.Tensor:
        """model.cross_attn's math (eval mode, no mask) over M move queries
        (e, (M, C)) and one position's 225 board tokens (board0, (225, C))."""
        return cross_attention_2d(
            self.q_proj, self.k_proj, self.v_proj, self.attn_out, self.num_heads, e, board0
        )

    def value(
        self, board: torch.Tensor, g: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """board (1, 225, C), g (1, 3C), e (M, C) -> attended (M, C) and the
        (wld (M, 3), score_diff (M, 2), planes (M, num_planes, 225)) heads. The
        plane head builds the full footprint distribution, then reduces it to the
        per-cell anchor marginal the evidence path consumes (footprint_cell_
        marginal), so the served planes stay (M, num_planes, 225)."""
        attended = self._cross_attention(e, board[0])
        hidden = F.relu(self.head_attended(attended) + self.head_g(g))
        out = self.head_out(hidden)  # (M, 5)
        wld = out[:, :3]
        score_diff = torch.cat([out[:, 3:4], F.softplus(out[:, 4:5]) + 1e-3], dim=1)
        plane_q = self.plane_attended(attended) + self.plane_g(g)  # (M, num_planes*slots*C)
        plane_q = plane_q.view(-1, self.num_planes, SLOTS_PER_CELL, self.c)
        anchored = torch.einsum("mhsc,nc->mhns", plane_q, board[0])  # (M, num_planes, N, slots)
        anchored = anchored.reshape(-1, self.num_planes, board.shape[1] * SLOTS_PER_CELL)
        catch = (self.plane_catch_attended(attended) + self.plane_catch_g(g)).view(
            -1, self.num_planes, CATCH_ALL
        )
        footprint_logits = torch.cat([anchored, catch], dim=-1)  # (M, num_planes, NUM_CLASSES)
        planes = footprint_cell_marginal(footprint_logits).flatten(2)  # (M, num_planes, 225)
        return attended, wld, score_diff, planes

    def gain(self, attended: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """The proves-best expected gain (M,) >= 0 off the same fused vector."""
        hidden = F.relu(self.pb_attended(attended) + self.pb_g(g))
        return F.softplus(self.pb_out(hidden)).squeeze(1)


class ProposalCacheExportModel(nn.Module):
    """The `move_proposal_cache` forward: trunk + move encodings + evidence-free
    scoring, exposing the cache tensors alongside the plain predictions.

    Board inputs are (1, ...) and the M move rows query the single position's
    board tokens directly -- MoveSetEvalModel.encode_board / encode_moves /
    plain score_moves at P=1, where the padded (P, maxK, C) grid degenerates.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        self.trunk = model.trunk
        self.board_pos_emb = model.board_pos_emb
        self.move_encoder = model.move_encoder
        self.heads = _ScoringHeads(model)

    def forward(
        self,
        input_spatial: torch.Tensor,  # (1, planes, 15, 15) f32
        input_scalar: torch.Tensor,  # (1, S) f32
        move_letters: torch.Tensor,  # (M, T) i32
        move_blanks: torch.Tensor,  # (M, T) u8
        move_squares: torch.Tensor,  # (M, T) i32
        move_tile_mask: torch.Tensor,  # (M, T) u8
        move_scalars: torch.Tensor,  # (M, 3) f32
    ) -> tuple[torch.Tensor, ...]:
        letters = move_letters.long()
        squares = move_squares.long()
        tile_mask = move_tile_mask.float()

        x, s = self.trunk(input_spatial, input_scalar)  # (1,C,15,15), (1,C)
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (1, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)  # (1, 3C)

        tile_board = board[0][squares]  # (M, T, C)
        tile_board = tile_board * move_scalars[:, 2].view(-1, 1, 1)  # is_play gate
        move_enc = self.move_encoder(letters, move_blanks, tile_mask, move_scalars, tile_board)

        _, wld, score_diff, planes = self.heads.value(board, g, move_enc)
        return board, g, move_enc, wld, score_diff, planes


class ProposalStepExportModel(nn.Module):
    """The `move_proposal_step` forward: encode the evidence tokens, fuse them
    into the cached board map, and re-score every candidate.

    The fusion stage is EvidenceFusion, run at P=1 with its two attentions
    re-expressed for the refitter (see module docstring). encode_tokens is
    reused verbatim -- it is already plain convs/Linears -- so only the two
    attentions and the padding masks are rebuilt here.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        self.fusion = model.evidence_fusion
        self.heads = _ScoringHeads(model)
        # Fusion self-attention: an nn.TransformerEncoderLayer whose inner MHA
        # carries the packed in_proj hazard -- split into plain q/k/v Linears.
        sa = self.fusion.self_attn.self_attn
        self.sa_num_heads = sa.num_heads
        self.sa_head_dim = sa.embed_dim // sa.num_heads
        self.sa_q, self.sa_k, self.sa_v = split_mha_qkv(sa)
        self.sa_out = sa.out_proj  # plain Linear

    def _self_attention(self, tokens: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        """The TransformerEncoderLayer (post-norm, dropout 0) over E evidence
        tokens (1, E, C), padded keys suppressed by the additive `key_bias`
        (1, E)."""
        h, d = self.sa_num_heads, self.sa_head_dim
        q = self.sa_q(tokens).view(1, -1, h, d).transpose(1, 2)  # (1, H, E, d)
        k = self.sa_k(tokens).view(1, -1, h, d).transpose(1, 2)
        v = self.sa_v(tokens).view(1, -1, h, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) * d**-0.5 + key_bias[:, None, None, :]
        attn = torch.softmax(logits, dim=-1)  # (1, H, E, E)
        ctx = (attn @ v).transpose(1, 2).reshape(1, -1, h * d)  # (1, E, C)
        sa = self.sa_out(ctx)
        layer = self.fusion.self_attn
        t = layer.norm1(tokens + sa)
        ff = layer.linear2(F.relu(layer.linear1(t)))
        return layer.norm2(t + ff)

    def _fuse(
        self,
        board: torch.Tensor,
        g: torch.Tensor,
        tokens: torch.Tensor,
        spatial_feats: torch.Tensor,
        m: torch.Tensor,
        has_ev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EvidenceFusion.forward at P=1: evidence self-attention, board tokens
        cross-attend into the set, and the conditioned (board, g). `m` (1, E) is
        the float evidence mask, `has_ev` (1, 1) marks positions with any real
        token."""
        f = self.fusion
        # Self-attention: real keys unbiased, pad keys -inf, but an all-empty
        # set (has_ev == 0) keeps every key so the softmax stays finite (the row
        # is gated to zero downstream regardless).
        sa_key = (m - 1.0) * NEG_BIAS * has_ev  # (1, E)
        t = self._self_attention(tokens, sa_key)
        t = t * m.unsqueeze(-1)

        # Cross-attention: board queries into the evidence tokens, padded tokens
        # suppressed; an all-empty set is left attending to token 0.
        col0 = torch.maximum(m[:, :1], 1.0 - has_ev)  # (1, 1)
        attend = torch.cat([col0, m[:, 1:]], dim=1)  # (1, E)
        key_bias = (attend - 1.0) * NEG_BIAS  # (1, E)
        h, d = f.num_heads, f.head_dim
        q = f.q_norm(f.q_proj(board).view(1, board.shape[1], h, d)).transpose(1, 2)  # (1,H,225,d)
        k = f.k_norm(f.k_proj(t).view(1, -1, h, d)).transpose(1, 2)  # (1, H, E, d)
        v = f.v_proj(t).view(1, -1, h, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / (d**0.5) + key_bias[:, None, None, :]
        weights = torch.softmax(logits, dim=-1)  # (1, H, 225, E)
        attended = (weights @ v).transpose(1, 2).reshape(1, board.shape[1], self.heads.c)
        wmean = weights.mean(dim=1)  # (1, 225, E)

        local = torch.einsum("pne,pedn->pnd", wmean, spatial_feats)  # (1, 225, d_spatial)
        delta = f.out_proj(f.attended_norm(attended)) + f.spatial_out(f.local_norm(local))
        denom = m.sum(dim=1).clamp(min=1)  # (1,)
        pooled = f.pooled_norm(t.sum(dim=1) / denom.unsqueeze(-1))  # (1, C)

        board = board + delta * has_ev.view(-1, 1, 1)
        g = g + f.summary_out(pooled) * has_ev.view(-1, 1)
        return board, g

    def forward(
        self,
        board: torch.Tensor,  # (1, 225, C) f32
        g: torch.Tensor,  # (1, 3C) f32
        move_enc: torch.Tensor,  # (M, C) f32
        ev_move_enc: torch.Tensor,  # (1, E, C) f32
        ev_obs_planes: torch.Tensor,  # (1, E, 9, 15, 15) f32
        ev_obs_scalars: torch.Tensor,  # (1, E, 11) f32
        ev_mask: torch.Tensor,  # (1, E) u8
    ) -> tuple[torch.Tensor, ...]:
        tokens, spatial_feats = self.fusion.encode_tokens(
            ev_move_enc, ev_obs_planes, ev_obs_scalars
        )
        m = ev_mask.float()  # (1, E) in {0, 1}
        has_ev = m.amax(dim=1, keepdim=True)  # (1, 1)
        board_c, g_c = self._fuse(board, g, tokens, spatial_feats, m, has_ev)
        attended, wld, score_diff, planes = self.heads.value(board_c, g_c, move_enc)
        gain = self.heads.gain(attended, g_c)
        return wld, score_diff, planes, gain


def _export(
    wrapper: nn.Module,
    dummies,
    input_names,
    output_names,
    dynamic_axes,
    path: Path,
    *,
    graph: str,
    opp_leave_input: bool,
    move_encoding_version: int,
    proposal_export_id: str,
    opset: int,
):
    """Trace `wrapper` to `path` atomically and stamp its metadata: the shared
    common keys, the per-graph architecture signature (the engine-plan cache
    key), the graph kind, the move-encoding version gate, and the
    proposal_export_id that ties a cache graph to the step graph exported from
    the same model. `dynamo=False`/`do_constant_folding=False` keep every weight
    a plain named initializer for the refitter (see the plain exporter)."""
    with atomic_output(path) as tmp_path, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            wrapper,
            dummies,
            str(tmp_path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
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
                "graph": graph,
                "move_encoding_version": str(move_encoding_version),
                "proposal_export_id": proposal_export_id,
            },
        )


def export_proposal_cache(
    model: MoveSetEvalModel,
    path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    *,
    opp_leave_input: bool,
    move_encoding_version: int,
    proposal_export_id: str,
    board_size: int = 15,
    opset: int = 17,
):
    """Trace and write the `move_proposal_cache` graph: board + M candidates ->
    board, g, move_enc, and the plain wld / score_diff / planes. M is the single
    dynamic axis ("moves")."""
    path = Path(path)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    wrapper = ProposalCacheExportModel(model).to(device)
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
    # The move inputs and the M-indexed outputs ride "moves"; board and g are
    # fixed (1, ...) and carry no dynamic axis.
    move_and_dyn = (
        "move_letters",
        "move_blanks",
        "move_squares",
        "move_tile_mask",
        "move_scalars",
        "move_enc",
        "wld",
        "score_diff",
        "planes",
    )
    dynamic_axes = {name: {0: "moves"} for name in move_and_dyn}
    _export(
        wrapper,
        dummies,
        CACHE_INPUT_NAMES,
        CACHE_OUTPUT_NAMES,
        dynamic_axes,
        path,
        graph=GRAPH_CACHE,
        opp_leave_input=opp_leave_input,
        move_encoding_version=move_encoding_version,
        proposal_export_id=proposal_export_id,
        opset=opset,
    )
    if was_training:
        model.train()


def export_proposal_step(
    model: MoveSetEvalModel,
    path: str | Path,
    *,
    opp_leave_input: bool,
    move_encoding_version: int,
    proposal_export_id: str,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    board_size: int = 15,
    opset: int = 17,
):
    """Trace and write the `move_proposal_step` graph: the cache tensors plus a
    padded width-`max_evidence` evidence set -> the conditioned wld / score_diff
    / planes and the proves-best gain. M ("moves") is the single dynamic axis;
    the evidence inputs are fixed-width leading-1 batches."""
    path = Path(path)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    wrapper = ProposalStepExportModel(model).to(device)
    wrapper.eval()
    c = model.cross_attn.embed_dim
    e = max_evidence
    cells = board_size * board_size

    dummy_m = 5
    dummies = (
        torch.zeros(1, cells, c, device=device),  # board
        torch.zeros(1, 3 * c, device=device),  # g
        torch.zeros(dummy_m, c, device=device),  # move_enc
        torch.zeros(1, e, c, device=device),  # ev_move_enc
        torch.zeros(1, e, NUM_EVIDENCE_PLANES, board_size, board_size, device=device),
        torch.zeros(1, e, NUM_EVIDENCE_SCALARS, device=device),
        torch.zeros(1, e, dtype=torch.uint8, device=device),  # ev_mask
    )
    dynamic_axes = {name: {0: "moves"} for name in ("move_enc", *STEP_OUTPUT_NAMES)}
    _export(
        wrapper,
        dummies,
        STEP_INPUT_NAMES,
        STEP_OUTPUT_NAMES,
        dynamic_axes,
        path,
        graph=GRAPH_STEP,
        opp_leave_input=opp_leave_input,
        move_encoding_version=move_encoding_version,
        proposal_export_id=proposal_export_id,
        opset=opset,
    )
    if was_training:
        model.train()


def proposal_export_id(model: MoveSetEvalModel) -> str:
    """The fingerprint tying a cache graph to its step graph: a WEIGHT-sensitive
    hash of the model (weight_fingerprint), identical for both wrappers exported
    from one in-memory model and different for any other checkpoint -- so a
    loader can reject a cache graph paired with a step graph from a different
    model, not just an architecturally different one (which the per-graph
    architecture signature already separates)."""
    return weight_fingerprint(model)
