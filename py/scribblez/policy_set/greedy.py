"""Form 1 of the policy-set model (docs/plans/deep_bot.md, "Form 1: greedy
selection, with cached steps"): a set is built one move at a time, each step
taking the candidate with the largest predicted gain over the moves already
chosen, gain(a | S) = E[max(0, v(a) - max_{s in S} v(s))]. Every set starts
from the seed, the position's static-equity top move: the steps add the moves
most likely to beat it, so every gain is a margin over a real move, never a
whole win probability (as the gain over an empty set would be).

A step conditions the board on the chosen moves and re-scores every
candidate. The chosen moves become tokens (their plain candidate vectors,
projected), and SetStage folds them into the board token map and the
summary. It is the evidence-fusion stage's set half
(evidence_fusion.EvidenceFusion): the tokens self-attend, the board tokens
cross-attend into them under QK-norm, and zero-initialized projections write
the result back, so an empty set leaves the plain model exactly. Chosen-move
tokens carry no sim observations, so the fusion's observation channels have
no counterpart here. The candidates then attend to the conditioned map
through the encoder's own attention and fuse (model.PolicySetEncoder), and the
gain head reads them.

Everything before the stage is computed once per position: the board trunk,
the candidate embeddings, and the plain candidate vectors the tokens come
from. A step reruns only the stage, the re-scoring and the gain head.

Training rows come by subset assembly: per position, `rows` random chosen
sets, each the seed plus moves drawn toward high value as deployment sets
are. A candidate's target is the least, over the set's members s, of its
expected improvement over s: E[max(0, mu_a - mu_s)], with mu_a - mu_s taken
as normal about the two moves' mean paired difference over the rollouts both
ran, with that mean's standard error. Each is an upper bound on the
improvement over the set's best, so the least is the tightest of them: a
move that sims as a member does gets about 0, however unlike the set's best
it is. A near-tie the sims left unresolved gets a target in proportion to how
unresolved it is; a clearly worse move about 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from scribblez.policy_set.metrics import SET_SIZES
from scribblez.policy_set.model import Encoded, PolicySetEncoder, to_grid

# The gain loss reads win probabilities in percentage points: Huber with this
# delta, so typical gains (a few points) fall in its quadratic range.
GAIN_SCALE = 100.0
GAIN_HUBER = 5.0
# Chosen sets are drawn with weights exp((v - v_best) / TEMPERATURE), plus
# FLOOR, so they favour moves within a few points of the best.
SUBSET_TEMPERATURE = 0.05
SUBSET_FLOOR = 1e-3
# The gain head's starting output bias: softplus(-4) is about 0.018, a gain of
# a couple of points, where most targets sit. At 0 a fresh head would predict
# 0.69 (69 points) for every candidate.
GAIN_BIAS_INIT = -4.0
# The largest chosen set a training row holds, the seed included: one less
# than the largest set the eval builds.
MAX_CHOSEN = max(SET_SIZES) - 1


def _zero_init(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


class SetStage(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.token_norm = nn.LayerNorm(channels)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=2 * channels,
            dropout=0.0,
            batch_first=True,
        )
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attended_norm = nn.LayerNorm(channels)
        self.pooled_norm = nn.LayerNorm(channels)
        self.out_proj = _zero_init(nn.Linear(channels, channels))
        self.summary_out = _zero_init(nn.Linear(channels, 3 * channels))

    def forward(
        self, board: torch.Tensor, g: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """board (R, 225, C), g (R, 3C), tokens (R, E, C), mask (R, E) bool ->
        the conditioned board and summary. A row with no chosen move passes
        through unchanged."""
        tokens = self.token_norm(tokens)
        # A row with every key masked would give NaN; empty rows are gated off
        # below anyway, so unmask one key for them.
        pad = ~mask
        pad = pad & ~pad.all(dim=1, keepdim=True)
        t = self.self_attn(tokens, src_key_padding_mask=pad) * mask.unsqueeze(-1)
        attended = self._cross_attention(board, t, ~pad)
        denom = mask.sum(dim=1).clamp(min=1).to(board.dtype)
        pooled = self.pooled_norm(t.sum(dim=1) / denom.unsqueeze(-1))
        chosen = mask.any(dim=1).to(board.dtype)
        board = board + self.out_proj(self.attended_norm(attended)) * chosen.view(-1, 1, 1)
        return board, g + self.summary_out(pooled) * chosen.view(-1, 1)

    def _cross_attention(
        self, board: torch.Tensor, tokens: torch.Tensor, attend_to: torch.Tensor
    ) -> torch.Tensor:
        """Board tokens attending over the chosen tokens -> (R, 225, C)."""
        r, n, c = board.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_norm(self.q_proj(board).view(r, n, h, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(tokens).view(r, -1, h, d)).transpose(1, 2)
        v = self.v_proj(tokens).view(r, -1, h, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)  # (R, H, 225, E)
        logits = logits.masked_fill(~attend_to[:, None, None, :], -torch.inf)
        return (F.softmax(logits, dim=-1) @ v).transpose(1, 2).reshape(r, n, c)


@dataclass
class SetRows:
    """R rows, each one position of a batch and a chosen set of its
    candidates, as global candidate indices padded to E."""

    position: torch.Tensor  # (R,) int64
    chosen: torch.Tensor  # (R, E) int64, 0 at padding
    mask: torch.Tensor  # (R, E) bool

    def to(self, device) -> SetRows:
        return SetRows(self.position.to(device), self.chosen.to(device), self.mask.to(device))


def make_rows(position: list[int], chosen: list[list[int]]) -> SetRows:
    width = max([len(c) for c in chosen] + [1])
    idx = torch.zeros(len(chosen), width, dtype=torch.int64)
    mask = torch.zeros(len(chosen), width, dtype=torch.bool)
    for r, c in enumerate(chosen):
        idx[r, : len(c)] = torch.tensor(c, dtype=torch.int64)
        mask[r, : len(c)] = True
    return SetRows(torch.tensor(position, dtype=torch.int64), idx, mask)


def row_candidates(
    pos_id: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every candidate of each row's position, rows in order: their global
    indices and their row ids (Mr,)."""
    counts = torch.bincount(pos_id, minlength=int(position.max()) + 1)
    starts = torch.cumsum(counts, 0) - counts
    per_row = counts[position]
    row_id = torch.repeat_interleave(torch.arange(len(position), device=pos_id.device), per_row)
    offset = torch.cumsum(per_row, 0) - per_row
    local = torch.arange(int(per_row.sum()), device=pos_id.device) - offset[row_id]
    return starts[position][row_id] + local, row_id


class GreedyModel(nn.Module):
    """Form 1: the encoder, the set stage and the gain head."""

    def __init__(self, encoder: PolicySetEncoder, num_heads: int):
        super().__init__()
        self.encoder = encoder
        channels = encoder.board_pos_emb.shape[1]
        self.token = nn.Linear(channels, channels)
        self.stage = SetStage(channels, num_heads)
        self.gain = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, 1)
        )
        nn.init.constant_(self.gain[-1].bias, GAIN_BIAS_INIT)

    def cache(self, batch: dict[str, torch.Tensor]) -> tuple[Encoded, torch.Tensor]:
        """What every step reuses: the encoder's outputs and the plain
        candidate vectors the chosen-move tokens come from."""
        enc = self.encoder.encode(batch)
        h = self.encoder.candidate_vectors(enc.e, enc.board, enc.g, batch["move_pos_id"])
        return enc, h

    def gains(
        self, enc: Encoded, h: torch.Tensor, pos_id: torch.Tensor, rows: SetRows
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each row's predicted gain for every candidate of its position, in
        win probability: (gains, global candidate indices, row ids), (Mr,)
        each."""
        tokens = self.token(h[rows.chosen]) * rows.mask.unsqueeze(-1)
        board, g = self.stage(enc.board[rows.position], enc.g[rows.position], tokens, rows.mask)
        move_idx, row_id = row_candidates(pos_id, rows.position)
        vectors = self.encoder.candidate_vectors(enc.e[move_idx], board, g, row_id)
        return F.softplus(self.gain(vectors)).squeeze(1), move_idx, row_id


def sample_rows(
    win_prob: np.ndarray,
    equity: np.ndarray,
    pos_id: np.ndarray,
    rows_per_position: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[list[int]]]:
    """Subset assembly's chosen sets: per position, `rows_per_position` sets,
    each its seed (the largest `equity`) and 0..MAX_CHOSEN-1 more of its
    candidates (never all of them), drawn without replacement toward high win
    probability."""
    positions, chosen = [], []
    starts = np.flatnonzero(np.r_[True, pos_id[1:] != pos_id[:-1]])
    ends = np.r_[starts[1:], len(pos_id)]
    for p, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        seed = int(np.argmax(equity[lo:hi]))
        v = win_prob[lo:hi].astype(np.float64)
        w = np.exp((v - v.max()) / SUBSET_TEMPERATURE) + SUBSET_FLOOR
        w[seed] = 0.0
        # Beside the seed, at most all but one of the other candidates.
        most = max(min(MAX_CHOSEN - 1, hi - lo - 2), 0)
        for _ in range(rows_per_position):
            size = int(rng.integers(0, most + 1))
            picked = rng.choice(hi - lo, size=size, replace=False, p=w / w.sum()) if size else []
            positions.append(p)
            chosen.append([int(lo + seed)] + [int(lo + i) for i in picked])
    return positions, chosen


def paired_difference(
    values: torch.Tensor, rollouts: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean of a's rollout values less b's over the rollouts both ran,
    and that mean's standard error, per pair: `values` (M, R) and `rollouts`
    (M,) as dataset.rollout_grid gives them, `a` and `b` candidate indices."""
    n = torch.minimum(rollouts[a], rollouts[b])
    shared = torch.arange(values.shape[1], device=values.device) < n.unsqueeze(1)
    diff = (values[a] - values[b]) * shared
    k = n.clamp(min=1).to(values.dtype)
    mean = diff.sum(dim=1) / k
    var = (diff.square().sum(dim=1) / k - mean.square()).clamp(min=0)
    return mean, (var / k).sqrt()


def expected_improvement(mean: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
    """E[max(0, X)] for X ~ N(mean, se^2); max(0, mean) where se is 0."""
    z = mean / se.clamp(min=1e-12)
    return mean * torch.special.ndtr(z) + se * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)


@torch.no_grad()
def gain_targets(batch: dict[str, torch.Tensor], rows: SetRows, move_idx, row_id) -> torch.Tensor:
    """Each row's candidates' least expected improvement over a member of
    the row's set, from their paired rollouts; 0 for the members themselves.
    Every row holds at least its seed."""
    values, rollouts = batch["rollout_values"].float(), batch["rollouts"]
    members, in_set = rows.chosen[row_id], rows.mask[row_id]  # (Mr, E)
    gain = torch.full(move_idx.shape, torch.inf, device=values.device)
    for e in range(members.shape[1]):
        ei = expected_improvement(*paired_difference(values, rollouts, move_idx, members[:, e]))
        gain = torch.where(in_set[:, e], torch.minimum(gain, ei), gain)
    chosen = ((members == move_idx.unsqueeze(1)) & in_set).any(dim=1)
    return torch.where(chosen, 0.0, gain)


def gain_loss(model: GreedyModel, batch: dict[str, torch.Tensor], rows: SetRows) -> torch.Tensor:
    """Huber on the gains, in percentage points, over every row's candidates."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        enc, h = model.cache(batch)
        gains, move_idx, row_id = model.gains(enc, h, batch["move_pos_id"], rows)
    target = gain_targets(batch, rows, move_idx, row_id)
    return F.huber_loss(
        gains.float() * GAIN_SCALE, target * GAIN_SCALE, delta=GAIN_HUBER, reduction="mean"
    )


@torch.no_grad()
def greedy_picks(model: GreedyModel, batch: dict[str, torch.Tensor], n: int) -> list[list[int]]:
    """Each position's set of up to `n`, as candidate indices within the
    position, built greedily: the seed, then at each step the unchosen
    candidate with the largest predicted gain."""
    pos_id = batch["move_pos_id"]
    device = pos_id.device
    num_positions = batch["input_spatial"].shape[0]
    sizes = torch.bincount(pos_id, minlength=num_positions)
    starts = (torch.cumsum(sizes, 0) - sizes).tolist()
    sizes = sizes.tolist()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        enc, h = model.cache(batch)
    equity, _ = to_grid(batch["move_equity"].float(), pos_id, num_positions, -torch.inf)
    seeds = equity.argmax(dim=1).tolist()
    chosen = [[starts[p] + seeds[p]] for p in range(num_positions)]
    for _ in range(min(n, max(sizes)) - 1):
        # One row per position, in order, so the rows' candidates are the
        # batch's own, in its order.
        rows = make_rows(list(range(num_positions)), chosen).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gains, _, _ = model.gains(enc, h, pos_id, rows)
        gains = gains.float()
        gains[torch.tensor([i for c in chosen for i in c], device=device)] = -torch.inf
        grid, _ = to_grid(gains, pos_id, num_positions, -torch.inf)
        best = grid.argmax(dim=1).tolist()
        for p in range(num_positions):
            if len(chosen[p]) < sizes[p]:
                chosen[p].append(starts[p] + best[p])
    return [[i - starts[p] for i in c] for p, c in enumerate(chosen)]
