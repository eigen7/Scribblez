"""The M1a reader (docs/plans/supreme_bot_m1a.md, PR 4): a causal transformer
over a row's tokens (scribblez.transfer_test.tokens) that answers each pick
query with its candidate's label.

The queries are appended after the padded context and share one attention
mask with it: a context token sees the context tokens up to itself, a query
sees the context prefix it was asked at and itself, and nothing sees a query.
Positions are rotary, over the token index; a query sits at its prefix
length, right after what it sees. Attention is grouped-query, through
FlexAttention, with a learned per-head bias between tokens of the same
candidate (SLOT_BIAS_MAX).

Every head answers as a correction to the frozen teacher's prior for the
query's candidate (prior_outputs), and the corrections start at zero, so an
untrained reader is the prior and training learns what the evidence adds.
Per query, the heads give:

    wld        win/draw/loss logits
    score      the score difference's mean and log standard deviation, in
               SCORE_SCALE units: the spread of a single game, not of the
               estimate
    expected   the expected score's (win + draw / 2) mean and log spread: an
               estimate and its uncertainty
    opp_next,  footprint-class logits of the opponent's reply and the root
    self_next  mover's next move
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from scribblez.position_eval.model import FOOTPRINT_CLASSES, PLACEMENT_HEAD_NAMES
from scribblez.transfer_test.rows import NO_SLOT
from scribblez.transfer_test.tokens import TokenBatch, TokenEncoder
from scribblez.transformer_tower import RMSNorm, apply_rope

ROPE_THETA = 10000.0
FOOTPRINT_HEADS = ("opp_next", "self_next")
FOOTPRINT_RANK = 64  # the footprint heads' low-rank bottleneck
# Each footprint head's teacher placement head, by index in the prior cache.
PRIOR_PLACEMENT = {h: PLACEMENT_HEAD_NAMES.index(f"{h}_placement") for h in FOOTPRINT_HEADS}
# Each attention head adds a fixed bias to its scores between tokens of the
# same candidate slot, spread from 0 to this across the heads: some heads lean
# toward their own candidate's probes and the rest read the whole context,
# and the model learns which to use. Without it, a query must learn to find
# its candidate's dozen probe outcomes among two thousand tokens from a
# near-uniform start, which it did not do in 3,000 steps even on a target that
# was exactly their mean. Fixed rather than learned: a learned bias's gradient
# is a reduction over every attention score, which at least halved the
# training speed.
SLOT_BIAS_MAX = 4.0
# The expected-score head's spread before any evidence: about the label noise
# of a 100-rollout label, the scale of what the reader must learn to resolve.
EXPECTED_LOG_SPREAD = math.log(0.03)

_flex_attention = torch.compile(flex_attention, dynamic=False)


@dataclass(frozen=True)
class ReaderConfig:
    width: int
    depth: int
    heads: int
    kv_heads: int
    teacher_width: int
    max_slots: int
    ffn_mult: int = 4
    activation_checkpointing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def rope_tables(positions: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables (B, S, 1, head_dim / 2) for token positions (B, S)."""
    pairs = torch.arange(head_dim // 2, device=positions.device, dtype=torch.float32)
    freqs = ROPE_THETA ** (-2 * pairs / head_dim)
    angles = positions.float()[..., None] * freqs
    return torch.cos(angles)[:, :, None], torch.sin(angles)[:, :, None]


@dataclass
class SequenceLayout:
    """What every attention layer needs about a batch's sequence, the context
    then the queries: rotary tables, the mask, and each token's candidate
    slot."""

    rope: tuple[torch.Tensor, torch.Tensor]
    mask: BlockMask
    slots: torch.Tensor  # (B, T + Q) int64, NO_SLOT for root tokens


def sequence_layout(b: TokenBatch, head_dim: int) -> SequenceLayout:
    return SequenceLayout(
        rope=rope_tables(positions(b), head_dim),
        mask=reader_mask(b),
        slots=torch.cat([b.slot, b.query_slot], dim=1),
    )


class Attention(nn.Module):
    def __init__(self, cfg: ReaderConfig):
        super().__init__()
        if cfg.width % cfg.heads or cfg.heads % cfg.kv_heads:
            raise ValueError(f"heads {cfg.heads} must divide width and be a multiple of kv_heads")
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.head_dim = cfg.width // cfg.heads
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.kv = nn.Linear(cfg.width, 2 * cfg.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.register_buffer("slot_bias", torch.linspace(0.0, SLOT_BIAS_MAX, cfg.heads))

    def forward(self, x: torch.Tensor, layout: SequenceLayout) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_norm(self.q(x).view(b, s, self.heads, self.head_dim))
        k, v = self.kv(x).view(b, s, 2, self.kv_heads, self.head_dim).unbind(2)
        q, k = apply_rope(q, *layout.rope), apply_rope(self.k_norm(k), *layout.rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, heads, S, head_dim)
        y = _flex_attention(
            q,
            k,
            v,
            score_mod=_same_slot_bias(self.slot_bias, layout.slots),
            block_mask=layout.mask,
            enable_gqa=True,
        )
        return self.out(y.transpose(1, 2).reshape(b, s, -1))


def _same_slot_bias(bias: torch.Tensor, slots: torch.Tensor):
    """A score_mod adding head h's `bias[h]` where a token and the key share a
    candidate slot (root tokens have none)."""

    def score_mod(score, b, h, qi, kvi):
        same = (slots[b, qi] == slots[b, kvi]) & (slots[b, qi] != NO_SLOT)
        return score + torch.where(same, bias[h], 0.0)

    return score_mod


class FeedForward(nn.Module):
    """SwiGLU at about `ffn_mult` times the width's parameters."""

    def __init__(self, cfg: ReaderConfig):
        super().__init__()
        hidden = 2 * cfg.ffn_mult * cfg.width // 3
        self.gate_up = nn.Linear(cfg.width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ReaderConfig):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.width), Attention(cfg)
        self.ffn_norm, self.ffn = RMSNorm(cfg.width), FeedForward(cfg)

    def forward(self, x: torch.Tensor, layout: SequenceLayout) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), layout)
        return x + self.ffn(self.ffn_norm(x))


class Tower(nn.Module):
    """The blocks and the final norm: the part of the reader with one shape
    per run, which the trainer compiles."""

    def __init__(self, cfg: ReaderConfig):
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = RMSNorm(cfg.width)
        self.checkpointing = cfg.activation_checkpointing

    def forward(self, x: torch.Tensor, layout: SequenceLayout) -> torch.Tensor:
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = checkpoint(block, x, layout, use_reentrant=False)
            else:
                x = block(x, layout)
        return self.norm(x)


def mask_mod_of(b: TokenBatch):
    """The FlexAttention mask_mod of the module docstring's mask, over the
    context's T tokens followed by the Q queries."""
    t = b.kind.shape[1]
    q = b.query_slot.shape[1]
    length = (~b.pad).sum(dim=1)
    prefix = b.query_prefix

    def mask_mod(bi, h, qi, kvi):
        is_query = qi >= t
        query = (qi - t).clamp(min=0, max=q - 1)
        sees_prefix = (kvi < prefix[bi, query]) | (kvi == qi)
        causal = (kvi <= qi) & (kvi < length[bi])
        return torch.where(is_query, sees_prefix, causal)

    return mask_mod


def reader_mask(b: TokenBatch) -> BlockMask:
    batch, t = b.kind.shape
    s = t + b.query_slot.shape[1]
    return create_block_mask(mask_mod_of(b), batch, None, s, s, device=b.kind.device)


def positions(b: TokenBatch) -> torch.Tensor:
    """(B, T + Q) rotary positions: the token index, and a query's prefix."""
    batch, t = b.kind.shape
    context = torch.arange(t, device=b.kind.device).expand(batch, t)
    return torch.cat([context, b.query_prefix], dim=1)


def _footprint_head(width: int) -> nn.Module:
    head = nn.Sequential(
        nn.Linear(width, FOOTPRINT_RANK), nn.Linear(FOOTPRINT_RANK, FOOTPRINT_CLASSES)
    )
    _zero(head[1])
    return head


def _zero(layer: nn.Linear):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def prior_outputs(b: TokenBatch) -> dict[str, torch.Tensor]:
    """The teacher's prior for each query's candidate, in the heads' terms
    (B, Q, ...): what the reader answers before it has learned to read."""
    value = b.candidate["prior_value"][b.query_candidate].float()
    wld, mean, sd = value[..., :3], value[..., 3], value[..., 4]
    expected = wld[..., 0] + 0.5 * wld[..., 1]
    placement = b.candidate["prior_placement"][b.query_candidate]
    return {
        "wld": torch.log(wld.clamp(min=1e-6)),
        "score": torch.stack([mean, torch.log(sd.clamp(min=1e-3))], dim=-1),
        "expected": torch.stack([expected, torch.full_like(expected, EXPECTED_LOG_SPREAD)], -1),
        **{
            h: torch.log_softmax(placement[..., PRIOR_PLACEMENT[h], :].float(), dim=-1)
            for h in FOOTPRINT_HEADS
        },
    }


class Reader(nn.Module):
    def __init__(self, cfg: ReaderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg.width, cfg.teacher_width, cfg.max_slots)
        self.tower = Tower(cfg)
        self.value = nn.Linear(cfg.width, 3 + 2 + 2)
        _zero(self.value)
        self.footprint = nn.ModuleDict(
            {name: _footprint_head(cfg.width) for name in FOOTPRINT_HEADS}
        )

    def forward(self, b: TokenBatch, tower=None) -> dict[str, torch.Tensor]:
        """The heads' outputs per query, each (B, Q, ...): the prior plus the
        reader's correction. `tower` runs the blocks in place of self.tower
        (the trainer passes the compiled one)."""
        context, queries = self.encoder(b)
        x = torch.cat([context, queries.to(context.dtype)], dim=1)
        run = self.tower if tower is None else tower
        h = run(x, sequence_layout(b, self.cfg.width // self.cfg.heads))[:, context.shape[1] :]
        value = self.value(h).float()
        correction = {
            "wld": value[..., 0:3],
            "score": value[..., 3:5],
            "expected": value[..., 5:7],
            **{name: head(h).float() for name, head in self.footprint.items()},
        }
        prior = prior_outputs(b)
        return {k: prior[k] + correction[k] for k in correction}
