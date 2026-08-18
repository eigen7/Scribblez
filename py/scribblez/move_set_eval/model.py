"""Move set evaluation model: score a variable-size candidate set in one pass.

The model encodes the pre-move board once with the shared SpatialTrunk (the same
front end as the position evaluation model), then scores every candidate move
against that one encoding via cross-attention -- so the expensive board encode
is amortized across all candidates, not repeated per move (docs/roadmap.md,
track A).

Shapes follow the dataset's flattened, no-padding batch: P positions supply the
board inputs, and M candidate moves (concatenated across those positions) each
carry a `pos_id` into [0, P). Each move:

  * is embedded from its placed tiles (glyph + square embeddings, masked-mean
    pooled) fused with its move-level scalars;
  * cross-attends into its own position's board tokens (the 15x15 trunk feature
    map flattened to 225 tokens, plus a learned square positional embedding);
  * is fused with its position's global summary and projected to the five
    per-move targets: a WLD distribution (3 logits) and the score-diff
    (mean, std), matching the teacher readouts stored in the .mset sidecar;
  * additionally decodes the teacher's four placement planes for its own
    post-move state (roadmap item 1): the fused per-move vector is projected
    to one query per plane head and scored against the 225 board tokens, so a
    single per-move vector yields four (15, 15) maps without any per-move
    spatial decoding.

The heads predict what the teacher position evaluation model would output for
the candidate's post-move state, from the mover's POV -- this model is a
distillation of the position evaluation model over the candidate set.

The forward optionally conditions the scoring on a sim-evidence set (roadmap
item 2): the EvidenceFusion stage rewrites the board token map and position
summary between the trunk and the scoring machinery, which then reads the
conditioned map exactly as it reads the plain one. The staged methods
(encode_board / encode_moves / encode_evidence / score_moves) expose the
deployment loop's caching split -- everything up to the fusion is computed
once per decision point, and forward() composed of them is bit-identical to
the staged path.

docs/model_architectures.md diagrams this network; any change to the
architecture belongs in the same commit as the corresponding change there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.evidence_fusion import EvidenceFusion, EvidenceInputs
from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

from .moves import move_encoding_dims
from .targets import PLANE_NAMES


class MoveEncoder(nn.Module):
    """Embeds each candidate move into a query vector for cross-attention.

    The move-input layout (letter vocabulary, scalar count) is the engine's,
    queried via move_encoding_dims so the embeddings match the encoder that
    produces the inputs. A placed tile is represented by its letter embedding, a
    blank-flag embedding (so a natural letter and its blank twin share letter
    semantics), and the board token at the square it lands on -- gathered from
    the trunk's board map by the caller, which ties a move's position directly to
    the board's own representation of that square.
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
        (M, T, C) -- the board features at each placed tile's square -> (M, C)."""
        tile_tok = self.letter_emb(letters) + self.blank_emb(blanks.long()) + board_tokens
        tile_tok = tile_tok * tile_mask.unsqueeze(-1)
        denom = tile_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (M, 1)
        tile_pool = tile_tok.sum(dim=1) / denom  # (M, C), 0 for exchange/pass
        scalar_feat = self.scalar_mlp(scalars)  # (M, C)
        return self.fuse(torch.cat([tile_pool, scalar_feat], dim=1))


def _rank_within_position(pos_id: torch.Tensor, num_positions: int) -> tuple[torch.Tensor, int]:
    """Each move's index within its own position's candidate block, plus the
    largest block -- the (row, column) address that packs the flattened move set
    into a padded (P, maxK) grid.

    Assumes the dataset's layout: each position's moves are one contiguous run
    and the runs are in position order (move_set_eval.dataset._build_batch
    concatenates per-position blocks), so this is arithmetic on the block
    offsets rather than a sort.
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
    ):
        super().__init__()
        self.board_size = board_size
        self._backbone_frozen = False
        self.trunk = SpatialTrunk(
            spatial_planes, scalar_size, trunk_channels, num_blocks, lexicon_module=lexicon_module
        )
        _, num_scalars, letter_vocab, cells = move_encoding_dims()
        # The flattened board tokens lose their grid identity; a learned
        # per-square embedding restores it so a move query can attend to the
        # square it plays on, and so the tokens a move gathers at its footprint
        # carry position.
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
        # Placement-plane readout (PLANE_NAMES order): one C-wide query per
        # plane head from the same fused per-move vector, scored against the
        # board tokens -- cell (h, n) is query_h . board_token_n.
        self.num_planes = len(PLANE_NAMES)
        self.plane_proj = nn.Linear(head_in, self.num_planes * trunk_channels)

        self.evidence_fusion = EvidenceFusion(trunk_channels, num_heads=num_heads)
        # The proves-best head: gain >= 0 through a softplus.
        self.proves_best = nn.Sequential(
            nn.Linear(head_in, trunk_channels),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels, 1),
        )

    # The evidence path's own parameters -- the fusion stage and the proves-best
    # head -- against the backbone, everything else: the distilled student's.
    # The evidence trainer either freezes the backbone at the student's
    # checkpoint (freeze_backbone) or trains it jointly under a distillation
    # anchor at its own learning rate (backbone_parameters).
    EVIDENCE_MODULES = ("evidence_fusion", "proves_best")

    @classmethod
    def _is_evidence_param(cls, name: str) -> bool:
        return name.split(".", 1)[0] in cls.EVIDENCE_MODULES

    def freeze_backbone(self):
        """Freeze every parameter outside EVIDENCE_MODULES, and pin those
        modules to eval mode (the trunk's BatchNorm would otherwise switch to
        batch statistics and drift its running stats under train()). Empty-
        evidence exactness is then structural (the fusion's zero-init + hard
        gate), so the plain pass is the student's, bit for bit, whatever
        training does."""
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
        """Initialize from a distilled student's state dict. The student may
        predate the fusion stage and never has the proves-best head; those stay
        at their fresh (zero-init / random) values. Anything else missing, or
        anything unexpected, is a real mismatch and fails."""
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
        """Embed M flattened moves against the (plain) board token map -> (M, C).

        Gathers the board token at each placed tile's square from that move's
        own position, so the move encoder reads the board's representation of
        the squares it plays on (pad squares gather token 0, masked out).
        Exchange tiles carry letters but no squares (move_set_encoder.h):
        gating by is_play zeroes the square-(0,0) tokens their zero squares
        would otherwise gather, so a surrendered tile contributes its letter
        and blank embeddings alone.
        """
        t = squares.shape[1]
        tile_board = board[pos_id.unsqueeze(1).expand(-1, t), squares]  # (M, T, C)
        tile_board = tile_board * scalars[:, 2].view(-1, 1, 1)
        return self.move_encoder(letters, blanks, tile_mask, scalars, tile_board)

    def encode_evidence(
        self, board: torch.Tensor, evidence: EvidenceInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-candidate evidence tokens over the plain board map (the move
        encodings and token features must cache across loop iterations, so
        they read the map the trunk produced, never a conditioned one).
        Returns EvidenceFusion.encode_tokens' (tokens, spatial features)."""
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
        self, board: torch.Tensor, g: torch.Tensor, e: torch.Tensor, pos_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score M encoded moves (M, C) against the board map and summary --
        plain or evidence-conditioned, the machinery is identical.

        Returns {"wld": (M,3) logits, "score_diff": (M,2) = [mean, std>0],
        "planes": (M, num_planes, 225) per-cell logits, PLANE_NAMES order,
        "gain": (M,) the proves-best expected gain, >= 0}.
        """
        # Each move attends into its own position's board tokens. Grouping the
        # queries by position keeps the key/value set at one copy per position,
        # so the attention's W_k/W_v projections -- a function of the board
        # alone, and ~C times the arithmetic of the attention math they feed --
        # are amortized across candidates the same way the trunk is.
        rank, max_k = _rank_within_position(pos_id, board.shape[0])
        queries = board.new_zeros(board.shape[0], max_k, board.shape[2])  # (P, maxK, C)
        queries[pos_id, rank] = e
        # The per-square attention weights are not a model output, and asking
        # for them would both materialize a (P, maxK, 225) tensor -- the padded
        # grid's largest by far -- and hold the call off the fused kernels.
        attended, _ = self.cross_attn(queries, board, board, need_weights=False)
        attended = attended[pos_id, rank]  # (P, maxK, C) -> (M, C)

        head_in = torch.cat([attended, g[pos_id]], dim=1)  # (M, 4C)
        out = self.head(head_in)  # (M, 5)
        sd_mean = out[:, 3:4]
        sd_std = F.softplus(out[:, 4:5]) + 1e-3

        # Plane readout through the same padded grid as the cross-attention,
        # so the (P, 225, C) board tokens are contracted once per position
        # rather than gathered per move.
        c = board.shape[2]
        plane_q = board.new_zeros(board.shape[0], max_k, self.num_planes * c)
        plane_q[pos_id, rank] = self.plane_proj(head_in)
        plane_logits = torch.einsum(
            "pkhc,pnc->pkhn", plane_q.view(board.shape[0], max_k, self.num_planes, c), board
        )
        planes = plane_logits[pos_id, rank]  # (M, num_planes, 225)

        return {
            "wld": out[:, :3],
            "score_diff": torch.cat([sd_mean, sd_std], dim=1),
            "planes": planes,
            "gain": F.softplus(self.proves_best(head_in)).squeeze(1),
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
        """Encode P board positions and score the M flattened candidate moves,
        optionally conditioned on a per-position evidence set (score_moves'
        return contract). With `evidence` None -- or with a position's mask
        row empty -- the plain one-pass model, bit-exactly.
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
        if evidence is not None:
            tokens, spatial_feats = self.encode_evidence(board, evidence)
            board, g = self.evidence_fusion(board, g, tokens, spatial_feats, evidence.mask)
        return self.score_moves(board, g, e, move_pos_id)


def win_equity(probs: torch.Tensor) -> torch.Tensor:
    """Scalar move value from a WLD probability distribution (..., 3):
    P(win) + 0.5*P(draw), the expected game-point value used to rank candidates.
    This model and the teacher are ranked the same way for the recall metric;
    the caller softmaxes this model's logits, while the teacher targets are
    already probabilities."""
    return probs[..., 0] + 0.5 * probs[..., 1]


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_sd: float = 0.004,
    huber_delta_mean: float = 10.0,
    huber_delta_std: float = 10.0,
    lambda_planes: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Distillation loss over the flattened candidate set (mean over all moves).

    Args:
        outputs: forward() result. "wld" (M,3 logits), "score_diff" (M,2),
                 "planes" (M, num_planes, 225) logits.
        targets: "target_wld" (M,3) teacher probabilities, "target_score_diff"
                 (M,2) teacher [mean, std] in score points, and -- on a
                 plane-carrying corpus -- "target_planes" (M, num_planes, 225)
                 teacher probabilities.

    WLD is soft cross-entropy against the teacher distribution (distillation),
    the score-diff mean/std are Huber regressions in score points, and the
    planes are per-cell BCE against the teacher's (soft) probabilities. A
    batch without plane targets contributes a zero plane term (the plane head
    simply gets no gradient from it).
    """
    # Soft cross-entropy: -sum(teacher_prob * log_softmax(pred)).
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
        loss_planes = F.binary_cross_entropy_with_logits(
            outputs["planes"], targets["target_planes"]
        )
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
