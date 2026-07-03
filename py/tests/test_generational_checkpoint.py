"""Unit tests for the generational rolling checkpoint (restart state)."""

from types import SimpleNamespace

import torch
from scribblez.generational.checkpoint import GenerationalState, resume, save
from scribblez.paths import POST_MOVE_VALUE, TagPaths

_CPU = torch.device("cpu")


def _paths(tmp_path) -> TagPaths:
    return TagPaths("t", POST_MOVE_VALUE, mount_root=tmp_path)


def test_resume_without_checkpoint_is_fresh(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    state = resume(_paths(tmp_path), model, opt, _CPU)
    assert state == GenerationalState(0, 0, 0, 0)


def test_save_resume_roundtrip(tmp_path):
    paths = _paths(tmp_path)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # A real step, so model + optimizer state are nontrivial.
    model(torch.randn(3, 4)).sum().backward()
    opt.step()

    state = GenerationalState(
        rows_trained=512, generation_index=3, epoch_in_generation=2, checkpoint_index=7
    )
    save(paths, model, opt, state, SimpleNamespace(lr=0.1, tag="t"))

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.9)  # overwritten by the restore
    loaded = resume(paths, model2, opt2, _CPU)

    assert loaded == state
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert opt2.param_groups[0]["lr"] == 0.1  # optimizer state restored from the saved run
