"""The tilted-prior NLL (scribblez.leave_inference.loss) against a brute-force
reference: the streaming log-sum-exp, the implied readouts, the custom
backward, and the uniform prior it reduces to at eta = 0."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez.ffi import LeaveValueArrays  # noqa: E402
from scribblez.leave_inference import loss  # noqa: E402

# A full bag's tile counts (A..Z, blank).
FULL_BAG = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2]


@pytest.fixture(scope="module")
def table() -> loss.TableTensors:
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    return loss.TableTensors(LeaveValueArrays(), "cpu")


def _pools_and_leaves(table, k: int, n: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """`n` random pools (a full bag minus random tiles) and a length-k leave
    drawn from each."""
    pools, leaves = [], []
    bag = np.repeat(np.arange(27), FULL_BAG)
    for _ in range(n):
        unseen = rng.choice(bag, size=60, replace=False)
        pool = np.bincount(unseen, minlength=27)
        leave = np.bincount(rng.choice(unseen, size=k, replace=False), minlength=27)
        pools.append(pool)
        leaves.append(leave)
    return torch.tensor(np.array(pools)), torch.tensor(np.array(leaves))


def _reference(table, k, theta, a, b, pool, leave):
    """Brute-force NLL over every length-k row, differentiable by autograd."""
    rows = table.lengths[k]
    log_c = loss.log_choose_table(pool)  # (B, cols)
    cols = loss.one_hot_columns(rows.counts)  # (R, 27)
    base = log_c[:, cols].sum(dim=2)  # (B, R)
    presence = (rows.counts > 0).float()
    v = rows.value / loss.VALUE_SCALE
    lw = base + theta @ presence.T + a[:, None] * v - b[:, None] * v * v
    log_z = torch.logsumexp(lw, dim=1)
    row_ids = torch.from_numpy(table.row_index(k, leave.numpy()))
    return log_z - lw[torch.arange(len(row_ids)), row_ids], torch.softmax(lw, dim=1)


@pytest.mark.parametrize("k", [2, 3])
def test_streaming_matches_brute_force(table, k):
    rng = np.random.default_rng(k)
    pool, leave = _pools_and_leaves(table, k, 6, rng)
    torch.manual_seed(k)
    theta, a, b = torch.randn(6, 27), torch.randn(6), torch.rand(6)
    got = loss.tilted_nll(table, theta, a, b, leave, torch.full((6,), k), pool, chunk=500)
    want, p = _reference(table, k, theta, a, b, pool, leave)
    torch.testing.assert_close(got.nll, want, rtol=1e-5, atol=1e-4)
    presence = (table.lengths[k].counts > 0).float()
    torch.testing.assert_close(got.presence, p @ presence, rtol=1e-4, atol=1e-5)
    value = table.lengths[k].value
    torch.testing.assert_close(got.value_mean, p @ value, rtol=1e-4, atol=1e-3)


def test_custom_backward_matches_autograd(table):
    k = 3
    rng = np.random.default_rng(7)
    pool, leave = _pools_and_leaves(table, k, 5, rng)
    torch.manual_seed(7)
    params = [torch.randn(5, 27), torch.randn(5), torch.rand(5)]
    for t in params:
        t.requires_grad_()
    got = loss.tilted_nll(table, *params, leave, torch.full((5,), k), pool, chunk=700)
    got.nll.sum().backward()
    got_grads = [t.grad.clone() for t in params]
    for t in params:
        t.grad = None
    want, _ = _reference(table, k, *params, pool, leave)
    want.sum().backward()
    for g, t in zip(got_grads, params, strict=True):
        torch.testing.assert_close(g, t.grad, rtol=1e-4, atol=1e-5)


def test_mixed_lengths_in_one_batch(table):
    rng = np.random.default_rng(3)
    p2, l2 = _pools_and_leaves(table, 2, 3, rng)
    p4, l4 = _pools_and_leaves(table, 4, 3, rng)
    pool, leave = torch.cat([p2, p4]), torch.cat([l2, l4])
    length = torch.tensor([2, 2, 2, 4, 4, 4])
    theta = torch.randn(6, 27, requires_grad=True)
    a, b = torch.zeros(6), torch.zeros(6)
    got = loss.tilted_nll(table, theta, a, b, leave, length, pool)
    got.nll.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    for idx, k in ((slice(0, 3), 2), (slice(3, 6), 4)):
        want, _ = _reference(table, k, theta[idx], a[idx], b[idx], pool[idx], leave[idx])
        torch.testing.assert_close(got.nll[idx], want, rtol=1e-5, atol=1e-4)


def test_zero_tilt_is_the_uniform_prior(table):
    rng = np.random.default_rng(11)
    for k in (1, 3, 5):
        pool, leave = _pools_and_leaves(table, k, 4, rng)
        length = torch.full((4,), k)
        got = loss.tilted_nll(
            table, torch.zeros(4, 27), torch.zeros(4), torch.zeros(4), leave, length, pool
        )
        torch.testing.assert_close(
            got.nll, loss.uniform_nll(leave, length, pool), rtol=1e-5, atol=1e-4
        )


def test_counts_from_tiles():
    tiles = torch.tensor([[0, 0, 4, 26, -1, -1, -1], [-1] * 7])
    counts = loss.counts_from_tiles(tiles)
    assert counts[0, 0] == 2 and counts[0, 4] == 1 and counts[0, 26] == 1
    assert counts[0].sum() == 4 and counts[1].sum() == 0


def test_uniform_nll_is_the_hypergeometric_probability():
    # Hand-built pools, with tiles the pool holds once and tiles it lacks.
    pool = torch.zeros(3, 27, dtype=torch.long)
    pool[:, 0], pool[:, 1], pool[:, 4], pool[:, 26] = 3, 1, 2, 1
    pool[2, 2] = 5
    leave = torch.zeros(3, 27, dtype=torch.long)
    leave[0, 0], leave[0, 1] = 2, 1  # AAB
    leave[1, 4], leave[1, 26] = 1, 1  # E?
    leave[2, 2], leave[2, 0], leave[2, 4] = 3, 1, 2  # CCCAEE
    length = leave.sum(dim=1)
    want = []
    for p, ell in zip(pool.tolist(), leave.tolist(), strict=True):
        ways = math.prod(math.comb(n, c) for n, c in zip(p, ell, strict=True))
        want.append(-math.log(ways / math.comb(sum(p), sum(ell))))
    torch.testing.assert_close(
        loss.uniform_nll(leave, length, pool), torch.tensor(want), rtol=1e-5, atol=1e-5
    )
