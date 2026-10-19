"""The tilted-prior NLL of the opponent's true leave (docs/plans/deep_bot.md,
"The leave inference model"; docs/plans/deep_bot_v0.md, "The loss").

The model's 29 floats define, over the leave table's rows of the leave's
length,

    log p(l) = base(l; pool) + eta . T(l) - log Z(eta)

where base is the hypergeometric log prior Sum_c log C(n_c, k_c), eta = (theta,
a, b) and T(l) = (presence, v, -v^2). Over the engine's tile codes, base and
theta . presence together are one per-query table summed over a row's codes
(_scores), so a pass is one matmul against the rows' code one-hot.

That is an exponential family, so the NLL's gradient in eta is
E_p[T] - T(l_true): one streaming pass over the table,
in chunks of rows, yields log Z (a running log-sum-exp) and E_p[T] together,
and the backward needs nothing else. Memory stays at batch x chunk, never
batch x 737,311 rows.

The pass runs in full FP32: the trainers set matmul precision "high", which
would put the base matmul in TF32 and leave ~0.01 nats of noise per row.
"""

from dataclasses import dataclass

import numpy as np
import torch

from scribblez.ffi import LeaveValueArrays, format_layout

_LAYOUT = format_layout()["constants"]["leave_value_arrays"]
TILE_KINDS = 27
MAX_LENGTH = _LAYOUT["max_length"]
# The engine's tile codes (lexicon/leave_value_arrays.h): letter * COPIES +
# the copies of that letter earlier in the sorted leave. A leave is the set of
# its codes; a letter's code letter * COPIES marks its presence.
COPIES = _LAYOUT["copies_per_tile"]
CODES = _LAYOUT["codes"]
# A prior step for a copy the pool cannot supply: low enough that exp() of any
# row holding it is 0, finite so 0 * it stays 0 in the matmul.
INFEASIBLE = -1e4
# The value tilt acts on v / VALUE_SCALE, so a and b train at a sane scale;
# the export converts them to points (onnx_export.py).
VALUE_SCALE = 10.0
DEFAULT_CHUNK = 65536

# Random odd 64-bit coefficients for row keys: a leave's key is the wrapping
# sum of its codes' coefficients, so a true leave finds its row by binary
# search. Fixed, so keys agree across processes.
_KEY_COEFFS = np.random.default_rng(0x1EAFE).integers(1, 2**62, CODES, dtype=np.int64) | 1


def code_sets(counts: torch.Tensor) -> torch.Tensor:
    """(N, 27) per-tile counts -> (N, CODES) float, 1 at each code of the
    leave: the first c codes of a letter held c times."""
    copy = torch.arange(COPIES, device=counts.device)
    return (copy[None, None, :] < counts.long()[:, :, None]).flatten(1).float()


@dataclass
class LengthRows:
    """One leave length's rows on the device: per-slot codes and values. The
    one-hot over codes a pass multiplies by is expanded a chunk at a time."""

    codes: torch.Tensor  # (k, R) uint8
    value: torch.Tensor  # (R,) float32, in points
    keys: np.ndarray  # (R,) int64, sorted
    order: np.ndarray  # (R,) row index of each sorted key


def _one_hot(codes: torch.Tensor) -> torch.Tensor:
    """(k, C) codes -> (C, CODES) one-hot rows."""
    out = torch.zeros(codes.shape[1], CODES, device=codes.device)
    return out.scatter_(1, codes.T.long(), 1.0)


class TableTensors:
    """The engine's leave value arrays (ffi.LeaveValueArrays) on a device, per
    length, with a key index from a leave's codes to its row."""

    def __init__(self, arrays: LeaveValueArrays, device):
        self.klv_hash = arrays.content_hash
        self.lengths: dict[int, LengthRows] = {}
        for k in range(1, MAX_LENGTH + 1):
            codes, values = arrays.of_length(k)
            with np.errstate(over="ignore"):
                keys = _KEY_COEFFS[codes.astype(np.int64)].sum(axis=0)
            order = np.argsort(keys)
            if np.any(np.diff(keys[order]) == 0):
                raise ValueError(f"leave table length {k}: colliding row keys")
            self.lengths[k] = LengthRows(
                codes=torch.from_numpy(codes).to(device),
                value=torch.from_numpy(values).to(device),
                keys=keys[order],
                order=order,
            )

    def row_index(self, k: int, counts: np.ndarray) -> np.ndarray:
        """The rows of the length-k leaves with tile counts `counts` (B, 27)."""
        rows = self.lengths[k]
        sets = code_sets(torch.from_numpy(counts)).numpy().astype(np.int64)
        with np.errstate(over="ignore"):
            keys = (sets * _KEY_COEFFS).sum(axis=1)
        pos = np.searchsorted(rows.keys, keys)
        if np.any(pos >= len(rows.keys)) or np.any(
            rows.keys[np.minimum(pos, len(rows.keys) - 1)] != keys
        ):
            raise ValueError(f"a length-{k} leave is not in the leave table")
        return rows.order[pos]


def prior_steps(pool: torch.Tensor) -> torch.Tensor:
    """(B, CODES): each copy's step of the hypergeometric log prior, so a
    leave's log prior Sum_l log C(n_l, c_l) is the sum over its codes. Copy j
    of a letter the pool holds n of adds log((n - j) / (j + 1)), INFEASIBLE
    once j >= n."""
    n = pool.to(torch.float32)[:, :, None]  # (B, 27, 1)
    j = torch.arange(COPIES, device=pool.device, dtype=torch.float32)  # (COPIES,)
    feasible = j < n
    step = torch.log(torch.where(feasible, n - j, 1.0)) - torch.log(j + 1)
    return torch.where(feasible, step, torch.full_like(step, INFEASIBLE)).flatten(1)


def _scores(theta: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """(B, CODES): the per-query table a row sums over its codes, the prior
    steps plus each letter's weight theta on its first-copy code."""
    d = steps.clone()
    d[:, ::COPIES] += theta
    return d


def _tilt(a, b, value):
    """a v' - b v'^2 for every (batch row, table row), v' = value / VALUE_SCALE."""
    v = value / VALUE_SCALE
    return a[:, None] * v[None, :] - b[:, None] * (v * v)[None, :]


def _stream(scores, a, b, rows: LengthRows, chunk: int):
    """log Z and the expectations E[presence], E[v'], E[v'^2] under the tilted
    prior, by a running log-sum-exp over the table's rows in chunks."""
    batch = scores.shape[0]
    device = scores.device
    m = torch.full((batch,), -torch.inf, device=device)
    s = torch.zeros(batch, device=device)
    e_presence = torch.zeros(batch, TILE_KINDS, device=device)
    e_v = torch.zeros(batch, device=device)
    e_v2 = torch.zeros(batch, device=device)
    for start in range(0, rows.value.shape[0], chunk):
        sl = slice(start, start + chunk)
        one_hot, value = _one_hot(rows.codes[:, sl]), rows.value[sl]
        lw = scores @ one_hot.T + _tilt(a, b, value)  # (B, C)
        m_new = torch.maximum(m, lw.max(dim=1).values)
        scale = torch.exp(m - m_new)
        w = torch.exp(lw - m_new[:, None])
        v = value / VALUE_SCALE
        s = s * scale + w.sum(dim=1)
        e_presence = e_presence * scale[:, None] + w @ one_hot[:, ::COPIES]
        e_v = e_v * scale + w @ v
        e_v2 = e_v2 * scale + w @ (v * v)
        m = m_new
    return m + torch.log(s), e_presence / s[:, None], e_v / s, e_v2 / s


class _TiltedNll(torch.autograd.Function):
    """Per-row NLL of the true leaves, with the exponential-family backward."""

    @staticmethod
    def forward(ctx, theta, a, b, steps, true_codes, true_v, rows, chunk):
        scores = _scores(theta, steps)
        log_z, e_presence, e_v, e_v2 = _stream(scores, a, b, rows, chunk)
        true_log_w = (scores * true_codes).sum(dim=1) + a * true_v - b * true_v * true_v
        true_presence = true_codes[:, ::COPIES]
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
        return grad_theta, grad_a, grad_b, None, None, None, None, None


@dataclass
class TiltedResult:
    """Per-row results over the rows passed in: the NLL in nats, and the
    implied presence probabilities and leave-value mean and variance (points)."""

    nll: torch.Tensor  # (N,)
    presence: torch.Tensor  # (N, 27)
    value_mean: torch.Tensor  # (N,)
    value_var: torch.Tensor  # (N,)
    true_value: torch.Tensor  # (N,), the true leave's


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
    steps = prior_steps(pool)
    true_codes = code_sets(leave_counts)
    counts_np = leave_counts.detach().cpu().numpy().astype(np.int64)
    length_np = length.detach().cpu().numpy().astype(np.int64)
    nll_parts, idx_parts = [], []
    for k in np.unique(length_np):
        idx_np = np.flatnonzero(length_np == k)
        idx = torch.from_numpy(idx_np).to(device)
        rows = table.lengths[int(k)]
        row_ids = torch.from_numpy(table.row_index(int(k), counts_np[idx_np])).to(device)
        true_v = rows.value[row_ids] / VALUE_SCALE
        nll, e_presence, e_v, e_v2 = _TiltedNll.apply(
            theta[idx], a[idx], b[idx], steps[idx], true_codes[idx], true_v, rows, chunk
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
    log_prior = (prior_steps(pool) * code_sets(leave_counts)).sum(dim=1)
    total = pool.to(torch.float32).sum(dim=1)
    k = length.to(torch.float32)
    log_norm = torch.lgamma(total + 1) - torch.lgamma(k + 1) - torch.lgamma(total - k + 1)
    return log_norm - log_prior
