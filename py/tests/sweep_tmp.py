import pytest
from test_simulation import sim  # noqa: F401


@pytest.mark.parametrize("seed", range(100, 300))
def test_sweep(sim, seed):  # noqa: F811
    run = sim(seed)
    for _ in range(300):
        run.step()
