"""The comparison head (docs/plans/deep_bot_v0.md, "The comparison head"): the
opponent's leave decoded one tile at a time, in sorted tile order, each step
a 27-way softmax.

Sorted order makes a leave one sequence, so the sequence probabilities are
the leave probabilities. A prefix is a multiset, so a step reads the context
vector, the counts of the tiles chosen so far, and how many remain; nothing
else about the order matters.

Each step's mask keeps exactly the tiles that extend the prefix to some leave
the pool can supply: tiles no lower than the last one chosen, with a copy
left in the pool, and with enough tiles at or above it left for the rest of
the leave. Every masked-in step therefore leads to a valid leave, so the
probabilities over valid leaves sum to 1, as the tilted head's do over the
leave table.
"""

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.ffi import format_layout
from scribblez.leave_inference.loss import TILE_KINDS

MAX_LEAVE_LENGTH = format_layout()["constants"]["leave_inference"]["max_leave_length"]


class AutoregressiveHead(nn.Module):
    def __init__(self, context_channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_channels + TILE_KINDS + MAX_LEAVE_LENGTH, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, TILE_KINDS),
        )

    def step_logits(self, context, chosen, remaining):
        """(N, 27) logits for the next tile: `chosen` (N, 27) counts so far,
        `remaining` (N,) tiles still to choose, >= 1."""
        left = F.one_hot(remaining - 1, MAX_LEAVE_LENGTH).float()
        return self.net(torch.cat([context, chosen.float(), left], dim=1))


def step_mask(pool, chosen, last, remaining):
    """(N, 27) bool: the tiles that extend each prefix to a leave the pool can
    supply. `last` (N,) is the last tile chosen (-1 for none)."""
    tiles = torch.arange(TILE_KINDS, device=pool.device)
    available = pool - chosen  # (N, 27)
    # Tiles at or above t still available, after taking one t.
    at_or_above = available.flip(1).cumsum(1).flip(1)
    ordered = tiles[None, :] >= last[:, None]
    return ordered & (available > 0) & (at_or_above - 1 >= (remaining - 1)[:, None])


def sequence_nll(head: AutoregressiveHead, context, tiles, length, pool) -> torch.Tensor:
    """Per-row NLL of each true leave: `tiles` (N, 7) sorted tile indices
    padded with -1, of `length` (N,) 1..6, from `pool` (N, 27) counts. Steps
    run teacher-forced, all rows at once."""
    n = context.shape[0]
    device = context.device
    tiles = tiles.long()
    pool = pool.long()
    chosen = torch.zeros(n, TILE_KINDS, dtype=torch.long, device=device)
    last = torch.full((n,), -1, dtype=torch.long, device=device)
    nll = torch.zeros(n, device=device)
    for i in range(int(length.max())):
        active = length > i
        remaining = (length - i).clamp_min(1)
        logits = head.step_logits(context, chosen, remaining)
        # A finished row's step is discarded, but an all-masked softmax would
        # be NaN, and NaN survives the discard into the gradients.
        mask = step_mask(pool, chosen, last, remaining) | ~active[:, None]
        log_p = torch.log_softmax(logits.float().masked_fill(~mask, -torch.inf), dim=1)
        target = tiles[:, i].clamp_min(0)
        step = -log_p.gather(1, target[:, None])[:, 0]
        nll = nll + torch.where(active, step, torch.zeros_like(step))
        took = F.one_hot(target, TILE_KINDS) * active[:, None]
        chosen = chosen + took
        last = torch.where(active, target, last)
    return nll
