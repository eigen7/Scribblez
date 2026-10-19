"""The comparison head (scribblez.leave_inference.autoregressive): its step
mask keeps exactly the extensions to suppliable leaves, so its probabilities
over every leave a pool can supply sum to 1; a row's NLL does not depend on
the other rows of its batch; and the model wires each head."""

import itertools

import pytest
import torch
from scribblez.leave_inference.autoregressive import AutoregressiveHead, sequence_nll
from scribblez.leave_inference.model import LeaveInferenceModel


def _leaves(pool: list[int], k: int) -> list[list[int]]:
    """Every size-k multiset drawable from `pool`, as sorted tile lists."""
    tiles = [t for t, n in enumerate(pool) for _ in range(n)]
    return sorted({tuple(c) for c in itertools.combinations(tiles, k)})


def test_probabilities_over_suppliable_leaves_sum_to_one():
    torch.manual_seed(0)
    head = AutoregressiveHead(context_channels=8, hidden=16)
    pool = [0] * 27
    pool[0], pool[4], pool[18], pool[26] = 3, 2, 1, 2  # A A A E E S ? ?
    for k in (1, 2, 3, 5):
        leaves = _leaves(pool, k)
        n = len(leaves)
        tiles = torch.full((n, 7), -1)
        for i, leave in enumerate(leaves):
            tiles[i, :k] = torch.tensor(leave)
        context = torch.randn(1, 8).expand(n, 8)
        nll = sequence_nll(head, context, tiles, torch.full((n,), k), torch.tensor([pool] * n))
        assert torch.isfinite(nll).all()
        torch.testing.assert_close(torch.exp(-nll).sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_mixed_lengths_score_each_row_as_alone():
    torch.manual_seed(1)
    head = AutoregressiveHead(context_channels=8, hidden=16)
    context = torch.randn(3, 8, requires_grad=True)
    tiles = torch.tensor([[0, -1, -1, -1, -1, -1, -1], [0, 0, 4, -1, -1, -1, -1], [4] * 6 + [-1]])
    pool = torch.zeros(3, 27, dtype=torch.long)
    pool[:, 0], pool[:, 4] = 3, 6
    length = torch.tensor([1, 3, 6])
    nll = sequence_nll(head, context, tiles, length, pool)
    for i in range(3):
        alone = sequence_nll(
            head, context[i : i + 1], tiles[i : i + 1], length[i : i + 1], pool[:1]
        )
        torch.testing.assert_close(nll[i : i + 1], alone)
    nll.sum().backward()
    assert torch.isfinite(context.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize(
    ("head", "keys"), [("tilted", {"theta", "a", "b"}), ("autoregressive", {"context"})]
)
def test_model_outputs_per_head(head, keys):
    model = LeaveInferenceModel(
        spatial_planes=4, scalar_size=5, trunk_channels=8, num_blocks=1, head=head
    )
    out = model(torch.zeros(2, 4, 15, 15), torch.zeros(2, 5))
    assert set(out) == keys


def test_unknown_head_is_refused():
    with pytest.raises(ValueError, match="unknown head"):
        LeaveInferenceModel(
            spatial_planes=4, scalar_size=5, trunk_channels=8, num_blocks=1, head="x"
        )
