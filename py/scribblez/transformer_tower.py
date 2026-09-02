"""KataGo-style transformer tower for the shared spatial trunk.

The conv tower in spatial_trunk.py reasons spatially through stacked 3x3
convolutions and periodically re-broadcasts board-global context through
global-pooling blocks. This tower replaces both with attention over the board's
cells as a token sequence, mirroring the trunk of KataGo's released transformer
nets (its NestedBottleneckTransformerBlock): each block projects the trunk
stream down to a narrower width, runs a few (self-attention, SwiGLU FFN) pairs
there, and projects back up onto the residual stream. Position enters through
2D rotary embeddings on the attention queries and keys, with per-head learnable
frequencies, so a head can be as local or as global as it learns to be.

The sequence may carry extra *register* tokens past the board cells -- content
the caller supplies (the position-eval model's tile-supply tokens) that every
attention layer sees alongside the cells. Registers have learnable 2D positions,
initialised just off the board, so the same rotary machinery covers them.

Training memory: each (attention, FFN) pair is activation-checkpointed --
autograd keeps only the pair's input and recomputes its ~dozen internal
activations during backward. Without it the 20 pairs of the production config
(10 blocks x inner_length 2, mid 192, 252 tokens) need ~25 GiB at batch 256 in
fp32; with it, ~5 GiB, for roughly a quarter more step time. Checkpointing is
skipped outside a training backward (eval, no-grad passes, ONNX tracing), where
nothing is saved for backward anyway.

Export discipline (onnx_export_util.py): the attention projections are
per-projection nn.Linears and RMSNorm is elementwise, so the legacy tracer
emits every weight as a named initializer. The fused
scaled_dot_product_attention between them carries no weights and decomposes
to plain matmul + softmax in the exported graph.

docs/model_architectures.md diagrams this tower; any change to it belongs in
the same commit as the corresponding change there.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TransformerConfig:
    """Shape of one nested-bottleneck block: the inner width, its attention head
    count (head dim = mid_channels / num_heads, even so RoPE can rotate channel
    pairs), the SwiGLU hidden width, and the (attention, FFN) pairs per block."""

    mid_channels: int
    num_heads: int
    ffn_channels: int
    inner_length: int = 2

    def __post_init__(self):
        if self.mid_channels % self.num_heads != 0:
            raise ValueError(
                f"mid_channels {self.mid_channels} not divisible by num_heads {self.num_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head dim {self.head_dim} must be even for RoPE")
        if self.inner_length < 1:
            raise ValueError(f"inner_length {self.inner_length} must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.mid_channels // self.num_heads


class RMSNorm(nn.Module):
    """Root-mean-square normalisation over the last dim with a learned per-channel
    gain, written out elementwise so it exports as plain ops."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


# Learnable 2D RoPE frequencies start log-uniform between one radian per cell and
# one per 50 cells, with a random sign per rotation pair (KataGo's initialisation).
ROPE_MAX_FREQ = 1.0
ROPE_MIN_FREQ = 1.0 / 50.0


def rope_init_freqs(num_heads: int, num_pairs: int) -> torch.Tensor:
    """(num_heads, num_pairs, 2) initial (omega_x, omega_y) per head and pair."""
    log_mag = torch.empty(num_heads, num_pairs, 2).uniform_(
        math.log(ROPE_MIN_FREQ), math.log(ROPE_MAX_FREQ)
    )
    sign = torch.randint(0, 2, (num_heads, num_pairs, 2)) * 2 - 1
    return torch.exp(log_mag) * sign


def rope_tables(
    pos_x: torch.Tensor, pos_y: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin rotation tables, each (S, num_heads, num_pairs), for token positions
    pos_x / pos_y (S,) under per-head frequencies (num_heads, num_pairs, 2):
    angle = omega_x * x + omega_y * y."""
    omega_x, omega_y = freqs[None, :, :, 0], freqs[None, :, :, 1]
    angles = pos_x[:, None, None] * omega_x + pos_y[:, None, None] * omega_y
    return torch.cos(angles), torch.sin(angles)


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each consecutive channel pair of t (B, S, H, D) by the (S, H, D/2)
    angle tables."""
    pairs = t.reshape(*t.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.flatten(-2)


class AttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention over the token sequence, with 2D RoPE on
    the queries and keys. forward returns the residual; the caller adds it."""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = RMSNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.rope_freqs = nn.Parameter(rope_init_freqs(num_heads, self.head_dim // 2))

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, S, C) -> (B, S, H, D)."""
        return t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        cos, sin = rope_tables(pos_x, pos_y, self.rope_freqs)
        q = apply_rope(self._heads(self.q_proj(xn)), cos, sin).transpose(1, 2)  # (B, H, S, D)
        k = apply_rope(self._heads(self.k_proj(xn)), cos, sin).transpose(1, 2)
        v = self._heads(self.v_proj(xn)).transpose(1, 2)
        # Fused attention never materializes the (B, H, S, S) score matrix, which
        # the explicit softmax(q @ k^T) form keeps for backward at every layer --
        # ~20 x 400 MB at batch 256 over the 252-token board+register sequence.
        context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)  # (B, S, C)
        return self.out_proj(context)


class SwiGluBlock(nn.Module):
    """Pre-norm SwiGLU feed-forward: W2 (silu(W1 x) * (Wg x)). forward returns the
    residual; the caller adds it."""

    def __init__(self, channels: int, ffn_channels: int):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.up = nn.Linear(channels, ffn_channels, bias=False)
        self.gate = nn.Linear(channels, ffn_channels, bias=False)
        self.down = nn.Linear(ffn_channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        return self.down(F.silu(self.up(xn)) * self.gate(xn))


class TransformerPair(nn.Module):
    """One (attention, FFN) pair, each on its own residual connection."""

    def __init__(self, channels: int, num_heads: int, ffn_channels: int):
        super().__init__()
        self.attention = AttentionBlock(channels, num_heads)
        self.ffn = SwiGluBlock(channels, ffn_channels)

    def forward(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, pos_x, pos_y)
        return x + self.ffn(x)


class NestedBottleneckTransformerBlock(nn.Module):
    """One tower block: RMSNorm -> ReLU -> Linear C -> C_mid, `inner_length`
    transformer pairs at C_mid, RMSNorm -> ReLU -> Linear C_mid -> C. The
    up-projection is zero-initialised, so every block starts as the identity on
    the trunk stream (KataGo's fixup-style init). forward returns the residual."""

    def __init__(self, channels: int, cfg: TransformerConfig):
        super().__init__()
        self.down_norm = RMSNorm(channels)
        self.down = nn.Linear(channels, cfg.mid_channels, bias=False)
        self.pairs = nn.ModuleList(
            TransformerPair(cfg.mid_channels, cfg.num_heads, cfg.ffn_channels)
            for _ in range(cfg.inner_length)
        )
        self.up_norm = RMSNorm(cfg.mid_channels)
        self.up = nn.Linear(cfg.mid_channels, channels, bias=False)
        nn.init.zeros_(self.up.weight)

    def _run_pair(
        self, pair: TransformerPair, out: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor
    ) -> torch.Tensor:
        """The pair's forward; under a training backward it is recomputed rather
        than having its internal activations saved (see the module docstring)."""
        if self.training and torch.is_grad_enabled():
            return checkpoint(pair, out, pos_x, pos_y, use_reentrant=False)
        return pair(out, pos_x, pos_y)

    def forward(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        out = self.down(F.relu(self.down_norm(x)))
        for pair in self.pairs:
            out = self._run_pair(pair, out, pos_x, pos_y)
        return self.up(F.relu(self.up_norm(out)))


# Register tokens start in one column three cells past the board's left edge,
# spread down its height, so no register coincides with a cell or another register.
REGISTER_INIT_X = -3.0


def register_init_positions(num_registers: int, board_size: int) -> torch.Tensor:
    """(num_registers, 2) initial (x, y) positions for the register tokens."""
    ys = torch.linspace(0.0, board_size - 1.0, num_registers)
    return torch.stack([torch.full_like(ys, REGISTER_INIT_X), ys], dim=-1)


class TransformerTower(nn.Module):
    """`num_blocks` nested-bottleneck blocks over a token sequence of the board's
    board_size² cells (row-major) followed by `num_registers` register tokens,
    then a final RMSNorm. Cells sit at their grid coordinates; registers at
    learnable positions."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        cfg: TransformerConfig,
        board_size: int,
        num_registers: int = 0,
    ):
        super().__init__()
        cell = torch.arange(board_size * board_size)
        self.register_buffer("cell_x", (cell % board_size).float(), persistent=False)
        self.register_buffer("cell_y", (cell // board_size).float(), persistent=False)
        self.register_pos = (
            nn.Parameter(register_init_positions(num_registers, board_size))
            if num_registers > 0
            else None
        )
        self.blocks = nn.ModuleList(
            NestedBottleneckTransformerBlock(channels, cfg) for _ in range(num_blocks)
        )
        self.final_norm = RMSNorm(channels)

    def _positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_x, pos_y), each (S,), for the cells then the registers."""
        if self.register_pos is None:
            return self.cell_x, self.cell_y
        pos_x = torch.cat([self.cell_x, self.register_pos[:, 0]])
        pos_y = torch.cat([self.cell_y, self.register_pos[:, 1]])
        return pos_x, pos_y

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, cells + registers, C) -> same shape."""
        pos_x, pos_y = self._positions()
        for block in self.blocks:
            tokens = tokens + block(tokens, pos_x, pos_y)
        return self.final_norm(tokens)
