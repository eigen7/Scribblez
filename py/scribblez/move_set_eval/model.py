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
    (mean, std), matching the teacher readouts stored in the .mset sidecar.

The heads predict what the teacher position evaluation model would output for
the candidate's post-move state, from the mover's POV -- this model is a
distillation of the position evaluation model over the candidate set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.spatial_trunk import SpatialTrunk, mean_max_pool

from .moves import move_encoding_dims


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
    ) -> dict[str, torch.Tensor]:
        """Encode P board positions and score the M flattened candidate moves.

        Returns {"wld": (M,3) logits, "score_diff": (M,2) = [mean, std>0]}.
        """
        x, s = self.trunk(input_spatial, input_scalar)  # (P,C,15,15), (P,C)
        board = x.flatten(2).transpose(1, 2) + self.board_pos_emb  # (P, 225, C)
        g = torch.cat([mean_max_pool(x), s], dim=1)  # (P, 3C)

        # Gather the board token at each placed tile's square from that move's
        # own position, so the move encoder reads the board's representation of
        # the squares it plays on (pad squares gather token 0, masked out).
        t = move_squares.shape[1]
        tile_board = board[move_pos_id.unsqueeze(1).expand(-1, t), move_squares]  # (M, T, C)
        e = self.move_encoder(
            move_letters, move_blanks, move_tile_mask, move_scalars, tile_board
        )  # (M, C)

        # Each move attends into its own position's board tokens. Grouping the
        # queries by position keeps the key/value set at one copy per position,
        # so the attention's W_k/W_v projections -- a function of the board
        # alone, and ~C times the arithmetic of the attention math they feed --
        # are amortized across candidates the same way the trunk is.
        rank, max_k = _rank_within_position(move_pos_id, board.shape[0])
        queries = board.new_zeros(board.shape[0], max_k, board.shape[2])  # (P, maxK, C)
        queries[move_pos_id, rank] = e
        attended, _ = self.cross_attn(queries, board, board)  # (P, maxK, C)
        attended = attended[move_pos_id, rank]  # (M, C)

        head_in = torch.cat([attended, g[move_pos_id]], dim=1)  # (M, 4C)
        out = self.head(head_in)  # (M, 5)
        sd_mean = out[:, 3:4]
        sd_std = F.softplus(out[:, 4:5]) + 1e-3
        return {"wld": out[:, :3], "score_diff": torch.cat([sd_mean, sd_std], dim=1)}


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
) -> dict[str, torch.Tensor]:
    """Distillation loss over the flattened candidate set (mean over all moves).

    Args:
        outputs: forward() result. "wld" (M,3 logits), "score_diff" (M,2).
        targets: "target_wld" (M,3) teacher probabilities, "target_score_diff"
                 (M,2) teacher [mean, std] in score points.

    WLD is soft cross-entropy against the teacher distribution (distillation),
    the score-diff mean/std are Huber regressions in score points.
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

    total = loss_wld + lambda_sd * loss_sd
    return {
        "total": total,
        "wld": loss_wld,
        "score_diff": loss_sd,
        "score_diff_mean": loss_sd_mean,
        "score_diff_std": loss_sd_std,
    }
