"""Form 2 of the policy-set model (docs/plans/deep_bot.md, "Form 2: a one-shot
slot decoder"). `slots` learned query slots, conditioned on the position's
summary, self-attend and cross-attend to the position's candidate vectors
(model.PolicySetEncoder), and each points to one candidate through a softmax
over its cosine similarities to them, times a learned scale. Bounded logits
keep the pointer's softmax, and its gradient, from running away as the
unscaled dot products did. Trained with the coverage loss over slot prefixes,
the first N slots serve as the set for any N up to `slots`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.policy_set.metrics import SET_SIZES
from scribblez.policy_set.model import PolicySetEncoder, mean_spread, to_grid

# Below this cover probability the coverage loss uses its first-order form.
NEARLY_MISSED = 1e-4
# log p floor in the coverage loss: finite at padding, and far below any
# real candidate's log-probability.
LOG_P_FLOOR = -1e4
# The pointer's logit scale: its start, and its cap. A logit is the scale times
# a cosine, so 10 spreads the candidates over 20 nats at the start, and the cap
# bounds the logits at +-50 however the scale is trained.
POINTER_SCALE_INIT = 10.0
POINTER_SCALE_MAX = 50.0


class SlotDecoder(nn.Module):
    def __init__(self, channels: int, slots: int, layers: int, num_heads: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, channels) * 0.02)
        self.g_proj = nn.Linear(3 * channels, channels)
        layer = nn.TransformerDecoderLayer(
            channels,
            num_heads,
            dim_feedforward=4 * channels,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.log_scale = nn.Parameter(torch.tensor(math.log(POINTER_SCALE_INIT)))

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=POINTER_SCALE_MAX)

    def forward(self, h: torch.Tensor, g: torch.Tensor, pos_id: torch.Tensor):
        """Candidate vectors (M, C), summaries (P, 3C) -> pointer logits
        (P, slots, maxK) in FP32, -inf at padding, and the padding mask
        (P, maxK)."""
        memory, pad = to_grid(h, pos_id, g.shape[0])
        tgt = self.slots.unsqueeze(0) + self.g_proj(g).unsqueeze(1)  # (P, slots, C)
        out = self.decoder(tgt, memory, memory_key_padding_mask=pad)
        q = F.normalize(self.query(out).float(), dim=-1)
        k = F.normalize(self.key(memory).float(), dim=-1)
        logits = self.scale() * torch.einsum("psc,pkc->psk", q, k)
        return logits.masked_fill(pad.unsqueeze(1), float("-inf")), pad


class SlotModel(nn.Module):
    """Form 2: the encoder and the slot decoder."""

    def __init__(self, encoder: PolicySetEncoder, slots: int, layers: int, num_heads: int):
        super().__init__()
        self.encoder = encoder
        channels = encoder.board_pos_emb.shape[1]
        self.decoder = SlotDecoder(channels, slots, layers, num_heads)

    def forward(self, batch: dict[str, torch.Tensor]):
        h, g = self.encoder(batch)
        return self.decoder(h, g, batch["move_pos_id"])


def coverage_loss(
    logits: torch.Tensor, q_grid: torch.Tensor, sizes: tuple[int, ...] | None = None
) -> dict[int, torch.Tensor]:
    """The coverage loss per set size N in `sizes` (default: the SET_SIZES the
    model's slots reach), each a mean over positions:
    L_N = -sum_a q(a) log(1 - prod_{k<=N} (1 - p_k(a))), the negative
    log-probability that the first N slots cover the best move. `logits`
    (P, slots, maxK), `q_grid` (P, maxK) with 0 at padding."""
    if sizes is None:
        sizes = tuple(n for n in SET_SIZES if n <= logits.shape[1])
    # Floored so padding (-inf) stays finite; q is 0 there.
    log_p = F.log_softmax(logits.float(), dim=-1).clamp(min=LOG_P_FLOOR)
    # log prod (1 - p_k), accumulated over slots; p clamped below 1 so that a
    # move no slot can miss stays finite.
    miss = torch.cumsum(torch.log1p(-log_p.exp().clamp(max=1 - 1e-6)), dim=1)
    cum_log_p = torch.logcumsumexp(log_p, dim=1)  # log sum_{k<=n} p_k
    out = {}
    for n in sizes:
        log_cover = _log_cover(miss[:, n - 1], cum_log_p[:, n - 1])  # (P, maxK)
        out[n] = -(q_grid * log_cover).sum(dim=1).mean()
    return out


def _log_cover(miss: torch.Tensor, log_p_sum: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(miss)), the log-probability some slot of the prefix picks
    the move. Where the prefix nearly misses it, log(-expm1(miss)) loses its
    gradient to rounding, so the first-order form log sum p_k stands in, off
    by a relative O(sum p_k) there."""
    nearly_missed = miss > -NEARLY_MISSED
    # The unused branch's input is kept finite: torch.where would otherwise
    # pass its NaN gradient through.
    exact = torch.log(-torch.expm1(torch.where(nearly_missed, -1.0, miss)))
    return torch.where(nearly_missed, log_p_sum, exact)


def slot_picks(logits: torch.Tensor, pad: torch.Tensor) -> list[list[int]]:
    """Each position's ordered set, as candidate indices within the position:
    slot k takes its most probable move not taken by an earlier slot."""
    picks = []
    for pos_logits, pos_pad in zip(logits.float().cpu(), pad.cpu(), strict=True):
        n_moves = int((~pos_pad).sum())
        taken: list[int] = []
        # A slot passes over at most the earlier slots' picks, so its top
        # `slots` candidates hold its choice.
        k = min(pos_logits.shape[0], n_moves)
        for order in torch.topk(pos_logits[:, :n_moves], k, dim=1).indices.tolist():
            taken.append(next(m for m in order if m not in taken))
            if len(taken) == n_moves:
                break
        picks.append(taken)
    return picks


def slot_forward(model: SlotModel, batch: dict[str, torch.Tensor]):
    """The slot model's pointer logits and padding mask, under autocast."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        return model(batch)


@torch.no_grad()
def slot_diagnostics(model: SlotModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The first slot's logits' spread over each position's candidates, and the
    pointer's scale."""
    logits, pad = slot_forward(model, batch)
    return {"spread": mean_spread(logits[:, 0], pad), "pointer_scale": model.decoder.scale()}


def slot_loss(
    logits: torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """The batch's coverage loss summed over the set sizes the slots reach,
    and per N."""
    num_positions = batch["input_spatial"].shape[0]
    q_grid, _ = to_grid(batch["q"].float(), batch["move_pos_id"], num_positions)
    by_n = coverage_loss(logits, q_grid)
    return sum(by_n.values()), by_n
