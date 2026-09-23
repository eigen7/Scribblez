"""Export the move proposal model as the two evidence-path ONNX graphs.

The move proposal model is a MoveSetEvalModel with the evidence fusion stage
and the proves-best head. Its deployment loop
(docs/plans/sim_residual_feedback.md) encodes the board, the candidate moves
and the evidence-free predictions once per turn; after each sim it conditions
on the grown evidence set and re-scores every candidate without re-running the
trunk. The two graphs mirror that split, and MoveSetEvalModel's staged methods:

  * `move_proposal_cache`, once per turn: P=1 board inputs plus M candidate
    rows -> the cache (board tokens, global summary, per-move encodings) and
    the evidence-free wld, score_diff and planes.
  * `move_proposal_step`, once per loop iteration: the cache tensors plus an
    evidence set padded to a fixed width E -> the conditioned wld, score_diff
    and proves-best gain. The step graph emits no planes: evidence tokens
    carry their candidate's evidence-free planes from the cache graph, and
    nothing reads a conditioned plane. At (M, 4 * SLOTS_PER_CELL, 225) floats
    it would be by far the largest output, allocated at the engine's row
    ceiling. The gain head's best-so-far input is computed in-graph
    (evidence_fusion.best_so_far), so the engine stages nothing for it.

In both graphs M ("moves") is the only dynamic axis; the evidence inputs have
a fixed leading-1 batch.

Both graphs follow the plain exporter's TensorRT-refit rules (onnx_export.py's
module docstring). Here the rebuilt attentions are the scoring
cross-attention and both fusion attentions. The fusion cross-attention is
already plain Linears but is rebuilt anyway, so its padding mask becomes an
additive float bias (NEG_BIAS) rather than a boolean masked_fill; boolean ops
are poorly supported on the ONNX/TensorRT path.

The step graph re-encodes the evidence tokens every iteration instead of taking
them from a third graph. That costs little next to the rollouts each step
schedules; caching per-candidate encodings is left to the engine runtime.
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.evidence_fusion import NUM_EVIDENCE_PLANES, NUM_EVIDENCE_SCALARS, best_so_far
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

from .model import MoveSetEvalModel, footprint_slot_planes
from .moves import move_encoding_dims
from .targets import PLANE_NAMES

# The padded evidence-set width E baked into the step graph as a fixed shape;
# the engine must stage exactly this many evidence rows. 64 is comfortably above
# the deployment sim budget.
DEFAULT_MAX_EVIDENCE = 64

# ONNX `graph` metadata values; must match kGraphMoveProposal* in
# engine/include/nn/onnx_metadata.h, which the C++ loader checks.
GRAPH_CACHE = "move_proposal_cache"
GRAPH_STEP = "move_proposal_step"

# A logit bias that softmaxes to exactly zero weight: exp(-1e9) underflows to
# 0.0, so an additive `(attend - 1) * NEG_BIAS` mask matches a boolean
# masked_fill(-inf) bit for bit while staying a plain float op.
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
STEP_OUTPUT_NAMES = ("wld", "score_diff", "gain")


class _ScoringHeads(nn.Module):
    """MoveSetEvalModel.score_moves at P=1, rebuilt for the refitter.

    The cache graph runs it over the plain board map, the step graph over the
    evidence-conditioned one. `value` returns the attended embeddings that
    `planes` and `gain` then read.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        mha = model.cross_attn
        c = mha.embed_dim
        self.c = c
        self.num_heads = mha.num_heads
        self.q_proj, self.k_proj, self.v_proj = split_mha_qkv(mha)
        self.attn_out = mha.out_proj

        # Each head's first Linear over cat([attended, g, ...]) is split at C.
        self.head_attended, self.head_g = split_concat_linear(model.head[0], c)
        self.head_out = model.head[2]
        self.num_planes = len(PLANE_NAMES)
        self.plane_attended, self.plane_g = split_concat_linear(model.plane_proj, c)
        self.plane_catch_attended, self.plane_catch_g = split_concat_linear(model.plane_catch, c)
        # pb_rest takes cat([g, best_so_far]).
        self.pb_attended, self.pb_rest = split_concat_linear(model.proves_best[0], c)
        self.pb_out = model.proves_best[2]

    def _cross_attention(self, e: torch.Tensor, board0: torch.Tensor) -> torch.Tensor:
        """e (M, C) move queries over board0 (225, C) -> (M, C)."""
        return cross_attention_2d(
            self.q_proj, self.k_proj, self.v_proj, self.attn_out, self.num_heads, e, board0
        )

    def value(
        self, board: torch.Tensor, g: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """board (1, 225, C), g (1, 3C), e (M, C) -> attended (M, C) and the
        (wld (M, 3), score_diff (M, 2)) heads."""
        attended = self._cross_attention(e, board[0])
        hidden = F.relu(self.head_attended(attended) + self.head_g(g))
        out = self.head_out(hidden)  # (M, 5)
        wld = out[:, :3]
        score_diff = torch.cat([out[:, 3:4], F.softplus(out[:, 4:5]) + 1e-3], dim=1)
        return attended, wld, score_diff

    def planes(self, attended: torch.Tensor, g: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """Footprint probabilities in the evidence-channel layout
        (footprint_slot_planes), (M, num_planes * SLOTS_PER_CELL, 225): exactly
        what the C++ staging copies into each evidence token's predicted
        block. Cache graph only."""
        plane_q = self.plane_attended(attended) + self.plane_g(g)  # (M, num_planes*slots*C)
        plane_q = plane_q.view(-1, self.num_planes, SLOTS_PER_CELL, self.c)
        anchored = torch.einsum("mhsc,nc->mhns", plane_q, board[0])  # (M, num_planes, N, slots)
        anchored = anchored.reshape(-1, self.num_planes, board.shape[1] * SLOTS_PER_CELL)
        catch = (self.plane_catch_attended(attended) + self.plane_catch_g(g)).view(
            -1, self.num_planes, CATCH_ALL
        )
        footprint_logits = torch.cat([anchored, catch], dim=-1)  # (M, num_planes, NUM_CLASSES)
        return footprint_slot_planes(footprint_logits).flatten(2)  # (M, planes*slots, 225)

    def gain(self, attended: torch.Tensor, g: torch.Tensor, best: torch.Tensor) -> torch.Tensor:
        """Proves-best expected gain (M,) >= 0; `best` (1, 1) is the evidence
        set's best-so-far."""
        hidden = F.relu(self.pb_attended(attended) + self.pb_rest(torch.cat([g, best], dim=1)))
        return F.softplus(self.pb_out(hidden)).squeeze(1)


class ProposalCacheExportModel(nn.Module):
    """The `move_proposal_cache` forward: MoveSetEvalModel's encode_board,
    encode_moves and plain score_moves at P=1, returning the cache tensors
    alongside the evidence-free predictions."""

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

        attended, wld, score_diff = self.heads.value(board, g, move_enc)
        planes = self.heads.planes(attended, g, board)
        return board, g, move_enc, wld, score_diff, planes


class ProposalStepExportModel(nn.Module):
    """The `move_proposal_step` forward: encode the evidence tokens, fuse them
    into the cached board map, and re-score every candidate.

    EvidenceFusion.encode_tokens is plain convs and Linears and is reused
    as is; only the fusion's two attentions and their padding masks are
    rebuilt here.
    """

    def __init__(self, model: MoveSetEvalModel):
        super().__init__()
        self.fusion = model.evidence_fusion
        self.heads = _ScoringHeads(model)
        # The fusion self-attention is a TransformerEncoderLayer; its inner MHA
        # has the packed in_proj.
        sa = self.fusion.self_attn.self_attn
        self.sa_num_heads = sa.num_heads
        self.sa_head_dim = sa.embed_dim // sa.num_heads
        self.sa_q, self.sa_k, self.sa_v = split_mha_qkv(sa)
        self.sa_out = sa.out_proj

    def _self_attention(self, tokens: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        """The fusion's TransformerEncoderLayer (post-norm, no dropout) over
        tokens (1, E, C), with pad keys suppressed by the additive `key_bias`
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
        """EvidenceFusion.forward at P=1 -> conditioned (board, g). `m` (1, E)
        is the float evidence mask; `has_ev` (1, 1) is 1 if any token is real."""
        f = self.fusion
        # An empty set (has_ev == 0) keeps every key so the softmax stays
        # finite; its output is gated to zero below regardless.
        sa_key = (m - 1.0) * NEG_BIAS * has_ev  # (1, E)
        t = self._self_attention(tokens, sa_key)
        t = t * m.unsqueeze(-1)

        # As in EvidenceFusion._cross_attention, an empty set attends to token 0.
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
        ev_obs_planes: torch.Tensor,  # (1, E, NUM_EVIDENCE_PLANES, 15, 15) f32
        ev_obs_scalars: torch.Tensor,  # (1, E, 11) f32
        ev_mask: torch.Tensor,  # (1, E) u8
    ) -> tuple[torch.Tensor, ...]:
        tokens, spatial_feats = self.fusion.encode_tokens(
            ev_move_enc, ev_obs_planes, ev_obs_scalars
        )
        m = ev_mask.float()  # (1, E) in {0, 1}
        has_ev = m.amax(dim=1, keepdim=True)  # (1, 1)
        board_c, g_c = self._fuse(board, g, tokens, spatial_feats, m, has_ev)
        attended, wld, score_diff = self.heads.value(board_c, g_c, move_enc)
        best = best_so_far(ev_obs_scalars, m).unsqueeze(1)  # (1, 1)
        gain = self.heads.gain(attended, g_c, best)
        return wld, score_diff, gain


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
    trained_max_evidence: int,
    opset: int,
):
    """Trace `wrapper` to `path` atomically and stamp its metadata.

    Beyond the keys the plain exporter writes, a proposal graph carries
    proposal_export_id, which ties a cache graph to its step graph, and
    trained_max_evidence, the widest evidence set the fusion stage trained on.
    The deployed agent's sim budget must respect the latter: the step graph
    pads to DEFAULT_MAX_EVIDENCE, but a wider set than the model trained on is
    out of distribution."""
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
                "trained_max_evidence": str(trained_max_evidence),
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
    trained_max_evidence: int,
    board_size: int = 15,
    opset: int = 17,
):
    """Write the `move_proposal_cache` graph."""
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
        trained_max_evidence=trained_max_evidence,
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
    trained_max_evidence: int,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    board_size: int = 15,
    opset: int = 17,
):
    """Write the `move_proposal_step` graph, padded to `max_evidence`."""
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
        trained_max_evidence=trained_max_evidence,
        opset=opset,
    )
    if was_training:
        model.train()


def export_proposal_pair(
    model: MoveSetEvalModel,
    cache_path: str | Path,
    step_path: str | Path,
    spatial_planes: int,
    scalar_size: int,
    *,
    opp_leave_input: bool,
    move_encoding_version: int,
    trained_max_evidence: int,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    board_size: int = 15,
):
    """Write both graphs of one model under one proposal_export_id.

    The step graph is written first. Tag ledgers and match dispatch key on the
    cache graph, and each write is atomic, so the pair is complete as soon as
    the cache graph is visible."""
    xid = proposal_export_id(model)
    export_proposal_step(
        model,
        step_path,
        opp_leave_input=opp_leave_input,
        move_encoding_version=move_encoding_version,
        proposal_export_id=xid,
        trained_max_evidence=trained_max_evidence,
        max_evidence=max_evidence,
        board_size=board_size,
    )
    export_proposal_cache(
        model,
        cache_path,
        spatial_planes,
        scalar_size,
        opp_leave_input=opp_leave_input,
        move_encoding_version=move_encoding_version,
        proposal_export_id=xid,
        trained_max_evidence=trained_max_evidence,
        board_size=board_size,
    )


def proposal_export_id(model: MoveSetEvalModel) -> str:
    """A hash of the model's weights, shared by the cache and step graphs of
    one export. It lets a loader reject a pair from different checkpoints of
    the same architecture, which the architecture signature cannot catch."""
    return weight_fingerprint(model)
