"""The tilted-prior NLL of the opponent's true leave (docs/plans/deep_bot.md,
"The leave inference model"; docs/plans/deep_bot_v0.md, "The loss").

The model's 29 floats define, over the leave table's rows of the leave's
length,

    log p(l) = base(l; pool) + eta . T(l) - log Z(eta)

where base is the hypergeometric log prior Sum_c log C(n_c, k_c), eta = (theta,
a, b) and T(l) = (presence, v, -v^2). That is an exponential family, so the
NLL's gradient in eta is E_p[T] - T(l_true): one streaming pass over the table,
in chunks of rows, yields log Z (a running log-sum-exp) and E_p[T] together,
and the backward needs nothing else. Memory stays at batch x chunk, never
batch x 737,311 rows.

The pass runs in full FP32: the trainers set matmul precision "high", which
would put the base matmul in TF32 and leave ~0.01 nats of noise per row.
"""

from dataclasses import dataclass

import numpy as np
import torch

from scribblez.ffi import LeaveValueArrays

TILE_KINDS = 27
# A leave holds at most this many of one tile.
MAX_COUNT = LeaveValueArrays.MAX_LENGTH
# One-hot columns per row: a (tile, count) pair each.
ONE_HOT_COLUMNS = TILE_KINDS * (MAX_COUNT + 1)
# log C(n, c) for a count the pool cannot supply: low enough that exp() of any
# row holding it is 0, finite so 0 * it stays 0 in the matmul.
INFEASIBLE = -1e4
# The value tilt acts on v / VALUE_SCALE, so a and b train at a sane scale;
# the export converts them to points (onnx_export.py).
VALUE_SCALE = 10.0
DEFAULT_CHUNK = 65536

# Random odd 64-bit coefficients for row keys: a leave's key is the wrapping
# sum of count * coefficient over tiles, so a true leave finds its row by
# binary search. Fixed, so keys agree across processes.
_KEY_COEFFS = np.random.default_rng(0x1EAFE).integers(1, 2**62, TILE_KINDS, dtype=np.int64) | 1


def _keys(counts: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (counts.astype(np.int64) * _KEY_COEFFS).sum(axis=1)


@dataclass
class LengthRows:
    """One leave length's table rows on the device. The one-hot and presence
    matrices a pass multiplies by are expanded from `counts` a chunk at a
    time: whole, the length-6 one-hot alone would be ~560 MB."""

    counts: torch.Tensor  # (R, 27) uint8
    value: torch.Tensor  # (R,) float32, in points
    keys: np.ndarray  # (R,) int64, sorted
    order: np.ndarray  # (R,) row index of each sorted key


def one_hot_columns(counts: torch.Tensor) -> torch.Tensor:
    """(R, 27) counts -> (R, 27) indices of their (tile, count) one-hot columns."""
    return torch.arange(TILE_KINDS, device=counts.device) * (MAX_COUNT + 1) + counts.long()


def _one_hot(counts: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(counts.shape[0], ONE_HOT_COLUMNS, device=counts.device)
    return out.scatter_(1, one_hot_columns(counts), 1.0)


def counts_of(tiles: np.ndarray, k: int) -> np.ndarray:
    """(R, 6) tile codes of length-k leaves -> (R, 27) uint8 per-tile counts."""
    rows = np.repeat(np.arange(len(tiles)), k)
    flat = rows * TILE_KINDS + tiles[:, :k].reshape(-1).astype(np.int64)
    return (
        np.bincount(flat, minlength=len(tiles) * TILE_KINDS)
        .reshape(-1, TILE_KINDS)
        .astype(np.uint8)
    )


class TableTensors:
    """The engine's leave value arrays (ffi.LeaveValueArrays) on a device, per
    length, as per-tile counts and values, with a key index from a leave's
    counts to its row."""

    def __init__(self, arrays: LeaveValueArrays, device):
        self.klv_hash = arrays.content_hash
        self.lengths: dict[int, LengthRows] = {}
        for k in range(1, LeaveValueArrays.MAX_LENGTH + 1):
            tiles, values = arrays.of_length(k)
            counts = counts_of(tiles, k)
            keys = _keys(counts)
            order = np.argsort(keys)
            if np.any(np.diff(keys[order]) == 0):
                raise ValueError(f"leave table length {k}: colliding row keys")
            self.lengths[k] = LengthRows(
                counts=torch.from_numpy(counts).to(device),
                value=torch.from_numpy(values).to(device),
                keys=keys[order],
                order=order,
            )

    def row_index(self, k: int, counts: np.ndarray) -> np.ndarray:
        """The rows of the length-k leaves with tile counts `counts` (B, 27)."""
        rows = self.lengths[k]
        keys = _keys(counts)
        pos = np.searchsorted(rows.keys, keys)
        if np.any(pos >= len(rows.keys)) or np.any(
            rows.keys[np.minimum(pos, len(rows.keys) - 1)] != keys
        ):
            raise ValueError(f"a length-{k} leave is not in the leave table")
        return rows.order[pos]


def log_choose_table(pool: torch.Tensor) -> torch.Tensor:
    """(B, ONE_HOT_COLUMNS): log C(n_tile, c) at column (tile, c), INFEASIBLE
    where c > n_tile."""
    n = pool.to(torch.float32)[:, :, None]  # (B, 27, 1)
    c = torch.arange(MAX_COUNT + 1, device=pool.device, dtype=torch.float32)  # (C,)
    feasible = c <= n
    nn_ = torch.where(feasible, n, c)  # avoid lgamma of negatives where infeasible
    log_c = torch.lgamma(nn_ + 1) - torch.lgamma(c + 1) - torch.lgamma(nn_ - c + 1)
    return torch.where(feasible, log_c, torch.full_like(log_c, INFEASIBLE)).flatten(1)


def _tilt(theta, a, b, presence, value):
    """theta . presence + a v' - b v'^2 for every (batch row, table row), with
    v' = value / VALUE_SCALE."""
    v = value / VALUE_SCALE
    return theta @ presence.T + a[:, None] * v[None, :] - b[:, None] * (v * v)[None, :]


def _stream(theta, a, b, log_c, rows: LengthRows, chunk: int):
    """log Z and the expectations E[presence], E[v'], E[v'^2] under the tilted
    prior, by a running log-sum-exp over the table's rows in chunks."""
    batch = theta.shape[0]
    device = theta.device
    m = torch.full((batch,), -torch.inf, device=device)
    s = torch.zeros(batch, device=device)
    e_presence = torch.zeros(batch, TILE_KINDS, device=device)
    e_v = torch.zeros(batch, device=device)
    e_v2 = torch.zeros(batch, device=device)
    for start in range(0, rows.value.shape[0], chunk):
        sl = slice(start, start + chunk)
        counts, value = rows.counts[sl], rows.value[sl]
        presence = (counts > 0).float()
        lw = log_c @ _one_hot(counts).T + _tilt(theta, a, b, presence, value)  # (B, C)
        m_new = torch.maximum(m, lw.max(dim=1).values)
        scale = torch.exp(m - m_new)
        w = torch.exp(lw - m_new[:, None])
        v = value / VALUE_SCALE
        s = s * scale + w.sum(dim=1)
        e_presence = e_presence * scale[:, None] + w @ presence
        e_v = e_v * scale + w @ v
        e_v2 = e_v2 * scale + w @ (v * v)
        m = m_new
    return m + torch.log(s), e_presence / s[:, None], e_v / s, e_v2 / s


class _TiltedNll(torch.autograd.Function):
    """Per-row NLL of the true leaves, with the exponential-family backward."""

    @staticmethod
    def forward(ctx, theta, a, b, log_c, true_base, true_presence, true_v, rows, chunk):
        log_z, e_presence, e_v, e_v2 = _stream(theta, a, b, log_c, rows, chunk)
        true_log_w = (
            true_base + (theta * true_presence).sum(dim=1) + a * true_v - b * true_v * true_v
        )
        ctx.save_for_backward(e_presence, e_v, e_v2, true_presence, true_v)
        ctx.mark_non_differentiable(e_presence, e_v, e_v2)
        return log_z - true_log_w, e_presence, e_v, e_v2

    @staticmethod
    def backward(ctx, grad_nll, *_):
        e_presence, e_v, e_v2, true_presence, true_v = ctx.saved_tensors
        g = grad_nll
        grad_theta = (e_presence - true_presence) * g[:, None]
        grad_a = (e_v - true_v) * g
        grad_b = (true_v * true_v - e_v2) * g
        return grad_theta, grad_a, grad_b, None, None, None, None, None, None


@dataclass
class TiltedResult:
    """Per-row results over the rows passed in: the NLL in nats, and the
    implied presence probabilities and leave-value mean and variance (points)."""

    nll: torch.Tensor  # (N,)
    presence: torch.Tensor | None = None  # (N, 27)
    value_mean: torch.Tensor | None = None  # (N,)
    value_var: torch.Tensor | None = None  # (N,)
    true_value: torch.Tensor | None = None  # (N,), the true leave's


def tilted_nll(
    table: TableTensors,
    theta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    leave_counts: torch.Tensor,
    length: torch.Tensor,
    pool: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
) -> TiltedResult:
    """The NLL of each row's true leave (`leave_counts` (N, 27), of `length`
    1..6) under the tilted prior over `pool` (N, 27). Rows are grouped by
    length, one table pass per length present. All rows must be valid."""
    with torch.autocast(device_type=theta.device.type, enabled=False):
        with _highest_matmul_precision():
            return _tilted_nll(
                table, theta.float(), a.float(), b.float(), leave_counts, length, pool, chunk
            )


def _tilted_nll(table, theta, a, b, leave_counts, length, pool, chunk) -> TiltedResult:
    n = theta.shape[0]
    device = theta.device
    out = TiltedResult(
        nll=torch.zeros(n, device=device),
        presence=torch.zeros(n, TILE_KINDS, device=device),
        value_mean=torch.zeros(n, device=device),
        value_var=torch.zeros(n, device=device),
        true_value=torch.zeros(n, device=device),
    )
    log_c = log_choose_table(pool)
    counts_np = leave_counts.detach().cpu().numpy().astype(np.int64)
    length_np = length.detach().cpu().numpy().astype(np.int64)
    nll_parts, idx_parts = [], []
    for k in np.unique(length_np):
        idx_np = np.flatnonzero(length_np == k)
        idx = torch.from_numpy(idx_np).to(device)
        rows = table.lengths[int(k)]
        row_ids = torch.from_numpy(table.row_index(int(k), counts_np[idx_np])).to(device)
        true_counts = rows.counts[row_ids]
        true_base = log_c[idx].gather(1, one_hot_columns(true_counts)).sum(dim=1)
        true_v = rows.value[row_ids] / VALUE_SCALE
        nll, e_presence, e_v, e_v2 = _TiltedNll.apply(
            theta[idx],
            a[idx],
            b[idx],
            log_c[idx],
            true_base,
            (true_counts > 0).float(),
            true_v,
            rows,
            chunk,
        )
        nll_parts.append(nll)
        idx_parts.append(idx)
        out.presence[idx] = e_presence
        out.value_mean[idx] = e_v * VALUE_SCALE
        out.value_var[idx] = (e_v2 - e_v * e_v).clamp_min(0) * VALUE_SCALE**2
        out.true_value[idx] = rows.value[row_ids]
    if nll_parts:
        # Scattered through an out-of-place index_put so autograd tracks it.
        out.nll = torch.zeros(n, device=device).index_put(
            (torch.cat(idx_parts),), torch.cat(nll_parts)
        )
    return out


class _highest_matmul_precision:
    def __enter__(self):
        self._saved = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def __exit__(self, *exc):
        torch.set_float32_matmul_precision(self._saved)


def counts_from_tiles(tiles: torch.Tensor) -> torch.Tensor:
    """(N, 7) sorted tile indices padded with -1 -> (N, 27) per-tile counts."""
    t = tiles.long()
    valid = t >= 0
    counts = torch.zeros(t.shape[0], TILE_KINDS, dtype=torch.long, device=t.device)
    counts.scatter_add_(1, t.clamp_min(0), valid.long())
    return counts


def uniform_nll(
    leave_counts: torch.Tensor, length: torch.Tensor, pool: torch.Tensor
) -> torch.Tensor:
    """The NLL of each true leave under the untilted prior: a size-`length`
    hypergeometric draw from `pool`."""
    log_c = log_choose_table(pool)
    log_prior = log_c.gather(1, one_hot_columns(leave_counts)).sum(dim=1)
    total = pool.to(torch.float32).sum(dim=1)
    k = length.to(torch.float32)
    log_norm = torch.lgamma(total + 1) - torch.lgamma(k + 1) - torch.lgamma(total - k + 1)
    return log_norm - log_prior
