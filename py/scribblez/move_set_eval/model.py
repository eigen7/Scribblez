"""Move set evaluation model: score a variable-size candidate set in one pass.

The model is a distillation of the position evaluation model (the teacher) over
a candidate set: for each candidate move it predicts what the teacher would
output for the resulting post-move state, from the mover's point of view. It
encodes the pre-move board once with the shared SpatialTrunk, then scores every
candidate against that one encoding by cross-attention, so the expensive board
encode is amortized over all candidates instead of repeated per move.

Batches are flattened with no padding: P positions supply the board inputs, and
M candidate moves (concatenated across positions) each carry a `pos_id` into
[0, P). Per move, the model:

  * embeds the placed tiles (letter + blank-flag embeddings plus the board
    token at the tile's square, masked-mean pooled) fused with the move's
    scalars;
  * cross-attends into its own position's 225 board tokens (the 15x15 trunk
    map flattened, plus a learned per-square embedding);
  * fuses the result with its position's global summary and reads out
    - a WLD distribution (3 logits) and the score-diff (mean, std), matching
      the teacher readouts stored in the .mset sidecar, and
    - one footprint-categorical placement distribution per teacher placement
      head (targets.PLANE_NAMES). These are dot products of per-move queries
      against the board tokens, so no per-move spatial decoder is needed.

The forward can optionally condition on a sim-evidence set (roadmap item 2):
the EvidenceFusion stage rewrites the board token map and position summary
between the trunk and the scoring, which reads the conditioned map exactly as
it would the plain one. The staged methods (encode_board, encode_moves,
encode_evidence, score_moves) mirror the deployment loop's caching split:
everything before the fusion is computed once per decision point. forward() is
their composition and is bit-identical to calling them in sequence.

docs/model_architectures.md diagrams this network; keep it in step with any
architecture change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.evidence_fusion import EvidenceFusion, EvidenceInputs, best_so_far
from scribblez.footprint_spatial import ANCHORED, CATCH_ALL, SIDE, SLOTS_PER_CELL
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool
from scribblez.supply_registers import TileSupplyRegisters
from scribblez.transformer_tower import TransformerConfig

from .moves import move_encoding_dims
from .targets import PLANE_NAMES

# MoveSetEvalModel.forward's positional inputs, in order: the board, then the
# flattened candidates. The mset and evidence datasets key their batch tensors
# by these names.
INPUT_KEYS = ("input_spatial", "input_scalar")
MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)


class MoveEncoder(nn.Module):
    """Embeds each candidate move into a query vector for cross-attention.

    A placed tile is the sum of its letter embedding, a blank-flag embedding
    (so a natural letter and its blank twin share the letter's meaning), and
    the board token at the square it lands on. The caller gathers that token
    from the trunk's map, which ties the move directly to the board's own
    representation of the squares it occupies.
    """

    def __init__(self, channels: int, letter_vocab: int, num_scalars: int):
        super().__init__()
        self.letter_emb = nn.Embedding(letter_vocab, channels, padding_idx=0)
        self.blank_emb = nn.Embedding(2, channels)  # 0 natural, 1 blank
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalars, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.fuse = nn.Linear(2 * channels, channels)

    def forward(
        self,
        letters: torch.Tensor,
        blanks: torch.Tensor,
        tile_mask: torch.Tensor,
        scalars: torch.Tensor,
        board_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """letters/blanks/tile_mask (M, T), scalars (M, S), board_tokens
        (M, T, C) at each placed tile's square -> (M, C)."""
        tile_tok = self.letter_emb(letters) + self.blank_emb(blanks.long()) + board_tokens
        tile_tok = tile_tok * tile_mask.unsqueeze(-1)
        denom = tile_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (M, 1)
        tile_pool = tile_tok.sum(dim=1) / denom  # (M, C), 0 for exchange/pass
        scalar_feat = self.scalar_mlp(scalars)  # (M, C)
        return self.fuse(torch.cat([tile_pool, scalar_feat], dim=1))


def _rank_within_position(pos_id: torch.Tensor, num_positions: int) -> tuple[torch.Tensor, int]:
    """Each move's index within its position's candidate block, plus the
    largest block size: the (row, column) address that packs the flattened move
    set into a padded (P, maxK) grid.

    Requires each position's moves to be one contiguous run, with the runs in
    position order (as dataset._build_batch lays them out), so this is offset
    arithmetic rather than a sort.
    """
    counts = torch.bincount(pos_id, minlength=num_positions)  # (P,)
    starts = torch.cumsum(counts, dim=0) - counts  # (P,)
    rank = torch.arange(pos_id.shape[0], device=pos_id.device) - starts[pos_id]  # (M,)
    return rank, int(counts.max())


class MoveSetEvalModel(nn.Module):
    """Board trunk + move encoder + single-pass cross-attention scoring."""

    def __init__(
        self,
        spatial_planes: int,
        scalar_size: int,
        trunk_channels: int = 192,
        num_blocks: int = 10,
        num_heads: int = 4,
        board_size: int = 15,
        lexicon_module: nn.Module | None = None,
        transformer: TransformerConfig | None = None,
    ):
        super().__init__()
        self.board_size = board_size
        self._backbone_frozen = False
        # The transformer tower carries the tile-supply register tokens, as the
        # teacher's does, so the placement readout can gate a square's
        # cross-checks on whether those tiles are still available
        # (supply_registers.py).
        self.trunk = SpatialTrunk(
            spatial_planes,
            scalar_size,
            trunk_channels,
            num_blocks,
            lexicon_module=lexicon_module,
            transformer=transformer,
            registers=(TileSupplyRegisters(trunk_channels, scalar_size) if transformer else None),
            board_size=board_size,
        )
        _, num_scalars, letter_vocab, cells = move_encoding_dims()
        # Flattening the board map loses grid identity. A learned per-square
        # embedding restores it, both for the attention keys and for the tokens
        # a move gathers at its own squares.
        self.board_pos_emb = nn.Parameter(torch.zeros(cells, trunk_channels))
        nn.init.normal_(self.board_pos_emb, std=0.02)

        self.move_encoder = MoveEncoder(trunk_channels, letter_vocab, num_scalars)
        self.cross_attn = nn.MultiheadAttention(
            trunk_channels, num_heads, dropout=0.0, batch_first=True
        )

        # Per-move head: the attended move embedding fused with its position's
        # global summary (mean+max board pooling plus the scalar projection, 3C).
        head_in = trunk_channels + 3 * trunk_channels
        self.head = nn.Sequential(
            nn.Linear(head_in, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, 5),  # [wld(3), sd_mean, sd_std]
        )
        # Placement readout, one footprint-categorical distribution per head in
        # PLANE_NAMES order. An anchored class factors as (board cell, slot),
        # with SLOTS_PER_CELL orientation/length slots per square (footprint.h),
        # so the fused per-move vector is projected to SLOTS_PER_CELL queries per
        # head and logit(h, cell, slot) = query(h, slot) . board_token(cell).
        # The two non-spatial catch-all classes (pass, not-win) come from a
        # small direct head.
        self.num_planes = len(PLANE_NAMES)
        self.plane_proj = nn.Linear(head_in, self.num_planes * SLOTS_PER_CELL * trunk_channels)
        self.plane_catch = nn.Linear(head_in, self.num_planes * CATCH_ALL)

        self.evidence_fusion = EvidenceFusion(trunk_channels, num_heads=num_heads)
        # The proves-best head predicts a gain >= 0 (softplus) from the fused
        # per-move vector plus best-so-far, the best sim value in the evidence
        # set (evidence_fusion.best_so_far). The gain label is measured from
        # best-so-far, so the head gets it directly rather than having to
        # reconstruct it from the mean-pooled evidence summary.
        self.proves_best = nn.Sequential(
            nn.Linear(head_in + 1, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, 1),
        )

    # Top-level modules owned by the evidence path. Everything else is the
    # backbone, i.e. the distilled student. The evidence trainer either freezes
    # the backbone at the student's weights (freeze_backbone) or trains it on
    # the sim-outcome loss at its own learning rate (backbone_parameters).
    EVIDENCE_MODULES = ("evidence_fusion", "proves_best")

    @classmethod
    def _is_evidence_param(cls, name: str) -> bool:
        return name.split(".", 1)[0] in cls.EVIDENCE_MODULES

    def freeze_backbone(self):
        """Freeze every parameter outside EVIDENCE_MODULES and pin the backbone
        modules to eval mode, which train() then preserves; otherwise the
        trunk's BatchNorm would use batch statistics and drift its running
        stats. Because the fusion stage is a hard no-op on an empty evidence
        set, the plain pass then stays the student's, bit for bit, however
        long the evidence path trains."""
        self._backbone_frozen = True
        for name, param in self.named_parameters():
            param.requires_grad = self._is_evidence_param(name)
        self.train(self.training)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            for name, module in self.named_children():
                if name not in self.EVIDENCE_MODULES:
                    module.eval()
        return self

    def evidence_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if self._is_evidence_param(n)]

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not self._is_evidence_param(n)]

    def load_student(self, state_dict: dict):
        """Initialize from a distilled student's state dict.

        The evidence modules keep their fresh initialization when absent. The
        student's proves-best head is always discarded: distillation never
        trains it, and an older student may carry it without the best-so-far
        input width. Any other missing or unexpected key raises."""
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("proves_best.")}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        stray = [k for k in missing if not self._is_evidence_param(k)]
        if stray or unexpected:
            raise ValueError(
                f"student checkpoint mismatch: missing {stray}, unexpected {unexpected}"
            )

    def encode_board(
        self, input_spatial: torch.Tensor, input_scalar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode P positions into board token maps (P, 225, C) and global
        summaries (P, 3C)."""
        x, s = self.trunk(input_spatial, input_scalar)  # (P,C,15,15), (P,C)
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (P, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)  # (P, 3C)
        return board, g

    def encode_moves(
        self,
        board: torch.Tensor,
        letters: torch.Tensor,
        blanks: torch.Tensor,
        squares: torch.Tensor,
        tile_mask: torch.Tensor,
        scalars: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> torch.Tensor:
        """Embed M flattened moves against the plain board token map -> (M, C)."""
        # Pad slots gather token 0 and are masked out by the move encoder.
        # Exchange tiles carry letters but square 0 (move_set_encoder.h), so the
        # is_play scalar zeroes their gathered tokens: a surrendered tile
        # contributes only its letter and blank embeddings.
        t = squares.shape[1]
        tile_board = board[pos_id.unsqueeze(1).expand(-1, t), squares]  # (M, T, C)
        tile_board = tile_board * scalars[:, 2].view(-1, 1, 1)  # scalars[:, 2] = is_play
        return self.move_encoder(letters, blanks, tile_mask, scalars, tile_board)

    def encode_evidence(
        self, board: torch.Tensor, evidence: EvidenceInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-candidate evidence tokens -> EvidenceFusion.encode_tokens'
        (tokens, spatial features).

        Reads the plain board map, never a conditioned one, so the tokens can be
        cached across the deployment loop's iterations."""
        p, k = evidence.mask.shape
        pos_id = torch.arange(p, device=board.device).repeat_interleave(k)
        move_enc = self.encode_moves(
            board,
            evidence.letters.flatten(0, 1),
            evidence.blanks.flatten(0, 1),
            evidence.squares.flatten(0, 1),
            evidence.tile_mask.flatten(0, 1),
            evidence.scalars.flatten(0, 1),
            pos_id,
        ).view(p, k, -1)
        return self.evidence_fusion.encode_tokens(
            move_enc, evidence.obs_planes, evidence.obs_scalars
        )

    def score_moves(
        self,
        board: torch.Tensor,
        g: torch.Tensor,
        e: torch.Tensor,
        pos_id: torch.Tensor,
        best_so_far: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score M encoded moves (M, C) against a board map and summary, plain
        or evidence-conditioned. `best_so_far` (P,) feeds only the proves-best
        head; None means an empty evidence set (0).

        Returns:
          "wld":        (M, 3) logits
          "score_diff": (M, 2) [mean, std > 0]
          "planes":     (M, num_planes, NUM_CLASSES) footprint logits, PLANE_NAMES order
          "gain":       (M,) proves-best expected gain, >= 0
        """
        # Queries are grouped by position into a padded (P, maxK) grid so the
        # key/value set is one copy per position. The W_k/W_v projections depend
        # only on the board and cost ~C times the attention math they feed, so
        # this amortizes them across candidates the same way the trunk is.
        rank, max_k = _rank_within_position(pos_id, board.shape[0])
        queries = board.new_zeros(board.shape[0], max_k, board.shape[2])  # (P, maxK, C)
        queries[pos_id, rank] = e
        # Returning the weights would materialize a (P, maxK, 225) tensor, the
        # largest in the grid, and keep the call off the fused kernels.
        attended, _ = self.cross_attn(queries, board, board, need_weights=False)
        attended = attended[pos_id, rank]  # (P, maxK, C) -> (M, C)

        head_in = torch.cat([attended, g[pos_id]], dim=1)  # (M, 4C)
        out = self.head(head_in)  # (M, 5)
        sd_mean = out[:, 3:4]
        sd_std = F.softplus(out[:, 4:5]) + 1e-3

        # The placement readout reuses the padded grid, so the board tokens are
        # contracted once per position rather than gathered per move. Logits are
        # laid out (head, cell, slot) so they flatten to the anchored footprint
        # classes (class = cell * slots + slot); the catch-all logits follow.
        c = board.shape[2]
        p, n = board.shape[0], board.shape[1]
        hs = self.num_planes * SLOTS_PER_CELL
        plane_q = board.new_zeros(p, max_k, hs * c)
        plane_q[pos_id, rank] = self.plane_proj(head_in)
        anchored = torch.einsum(
            "pkhsc,pnc->pkhns", plane_q.view(p, max_k, self.num_planes, SLOTS_PER_CELL, c), board
        )  # (P, max_k, num_planes, N, slots)
        anchored = anchored[pos_id, rank].reshape(-1, self.num_planes, n * SLOTS_PER_CELL)
        catch = self.plane_catch(head_in).view(-1, self.num_planes, CATCH_ALL)
        planes = torch.cat([anchored, catch], dim=-1)  # (M, num_planes, NUM_CLASSES)

        if best_so_far is None:
            best_so_far = g.new_zeros(p)
        gain_in = torch.cat([head_in, best_so_far[pos_id].to(head_in.dtype).unsqueeze(1)], dim=1)
        return {
            "wld": out[:, :3],
            "score_diff": torch.cat([sd_mean, sd_std], dim=1),
            "planes": planes,
            "gain": F.softplus(self.proves_best(gain_in)).squeeze(1),
        }

    def forward(
        self,
        input_spatial: torch.Tensor,
        input_scalar: torch.Tensor,
        move_letters: torch.Tensor,
        move_blanks: torch.Tensor,
        move_squares: torch.Tensor,
        move_tile_mask: torch.Tensor,
        move_scalars: torch.Tensor,
        move_pos_id: torch.Tensor,
        evidence: EvidenceInputs | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode P positions and score the M flattened candidates, optionally
        conditioned on a per-position evidence set; returns score_moves' dict.
        With `evidence` None, or for a position whose evidence mask row is
        empty, the result is bit-identical to the plain one-pass model.
        """
        board, g = self.encode_board(input_spatial, input_scalar)
        e = self.encode_moves(
            board,
            move_letters,
            move_blanks,
            move_squares,
            move_tile_mask,
            move_scalars,
            move_pos_id,
        )
        best = None
        if evidence is not None:
            tokens, spatial_feats = self.encode_evidence(board, evidence)
            board, g = self.evidence_fusion(board, g, tokens, spatial_feats, evidence.mask)
            best = best_so_far(evidence.obs_scalars, evidence.mask)
        return self.score_moves(board, g, e, move_pos_id, best)


def win_equity(probs: torch.Tensor) -> torch.Tensor:
    """P(win) + 0.5 * P(draw) from WLD probabilities (..., 3): the expected
    game points used to rank candidates, for both student and teacher. Takes
    probabilities, so the caller softmaxes the model's logits first."""
    return probs[..., 0] + 0.5 * probs[..., 1]


def footprint_cell_marginal(plane_logits: torch.Tensor) -> torch.Tensor:
    """Probability that the move is anchored at each cell, from footprint
    logits (..., num_planes, NUM_CLASSES) -> (..., num_planes, SIDE, SIDE).

    Display only: the Trajectories pane draws it beside the observed anchor
    marginal. It sums away the slot axis, so model inputs use
    footprint_slot_planes instead.
    """
    probs = F.softmax(plane_logits, dim=-1)[..., :ANCHORED]
    per_cell = probs.reshape(*probs.shape[:-1], SIDE * SIDE, SLOTS_PER_CELL).sum(-1)
    return per_cell.reshape(*per_cell.shape[:-1], SIDE, SIDE)


def footprint_slot_planes(plane_logits: torch.Tensor) -> torch.Tensor:
    """Footprint logits (..., num_planes, NUM_CLASSES) -> anchored-class
    probabilities as board channels (..., num_planes * SLOTS_PER_CELL, SIDE, SIDE),
    class (cell, slot) of head h at channel h * SLOTS_PER_CELL + slot. The
    softmax runs over all classes and the catch-alls are then dropped without
    renormalizing.

    This is the predicted block of the evidence-plane layout
    (evidence_fusion.EVIDENCE_PLANE_NAMES) and, flattened over SIDE*SIDE, the
    `planes` output of the proposal graphs. The ONNX export traces it, so keep
    it to plain reshape/permute arithmetic.
    """
    probs = F.softmax(plane_logits, dim=-1)[..., :ANCHORED]
    spatial = probs.reshape(*probs.shape[:-1], SIDE, SIDE, SLOTS_PER_CELL)
    # (..., H, R, C, S) -> (..., H, S, R, C), as an explicit non-negative
    # permutation: a negative axis in the traced Transpose is rejected by
    # ONNXRuntime's type inference.
    d = spatial.dim()
    perm = list(range(d - 4)) + [d - 4, d - 1, d - 3, d - 2]
    return spatial.permute(perm).flatten(-4, -3)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_sd: float = 0.004,
    huber_delta_mean: float = 10.0,
    huber_delta_std: float = 10.0,
    lambda_planes: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Distillation loss over the flattened candidate set, averaged over moves.

    `targets` holds the teacher's "target_wld" (M, 3) probabilities,
    "target_score_diff" (M, 2) [mean, std] in score points, and, on a corpus
    that stores planes, "target_planes" (M, num_planes, NUM_CLASSES).

    WLD and planes are soft cross-entropy against the teacher distributions;
    the score-diff mean and std are Huber regressions. The plane softmax is
    unmasked: illegal footprints are already zero in the teacher target, so the
    student learns to suppress them. Without plane targets the plane term is 0.
    """
    log_pred = F.log_softmax(outputs["wld"], dim=1)
    loss_wld = -(targets["target_wld"] * log_pred).sum(dim=1).mean()

    sd_mean = outputs["score_diff"][:, 0]
    sd_std = outputs["score_diff"][:, 1]
    t_mean = targets["target_score_diff"][:, 0]
    t_std = targets["target_score_diff"][:, 1]
    loss_sd_mean = F.huber_loss(sd_mean, t_mean, delta=huber_delta_mean)
    loss_sd_std = F.huber_loss(sd_std, t_std, delta=huber_delta_std)
    loss_sd = loss_sd_mean + loss_sd_std

    if "target_planes" in targets:
        log_pred_planes = F.log_softmax(outputs["planes"], dim=-1)
        loss_planes = -(targets["target_planes"] * log_pred_planes).sum(dim=-1).mean()
    else:
        loss_planes = outputs["wld"].new_zeros(())

    total = loss_wld + lambda_sd * loss_sd + lambda_planes * loss_planes
    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "score_diff_mean": loss_sd_mean,
        "score_diff_std": loss_sd_std,
        "planes": loss_planes,
    }
