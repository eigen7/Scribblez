"""Unit tests for the generational rolling checkpoint (restart state)."""

from dataclasses import asdict, dataclass

import torch
from scribblez.generational.checkpoint import GenerationalState, resume, save
from scribblez.paths import POSITION_EVAL, TagPaths

_CPU = torch.device("cpu")


def _paths(tmp_path) -> TagPaths:
    return TagPaths("t", POSITION_EVAL, mount_root=tmp_path)


def test_resume_without_checkpoint_is_fresh(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    state = resume(_paths(tmp_path), model, opt, _CPU)
    assert state == GenerationalState(0, 0)


def test_save_resume_roundtrip(tmp_path):
    paths = _paths(tmp_path)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # A real step, so model + optimizer state are nontrivial.
    model(torch.randn(3, 4)).sum().backward()
    opt.step()

    state = GenerationalState(rows_trained=512, generation_index=3)
    save(paths, model, opt, state, {"lr": 0.1, "tag": "t"})

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.9)  # overwritten by the restore
    loaded = resume(paths, model2, opt2, _CPU)

    assert loaded == state
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert opt2.param_groups[0]["lr"] == 0.1  # optimizer state restored from the saved run


@dataclass
class _ExtendedState(GenerationalState):
    """A trainer needing more cursor than the two shared fields."""

    settled_epochs: int = 0


def test_a_subclass_cursor_rides_along_and_the_base_one_stays_two_fields(tmp_path):
    """The checkpoint persists whatever the state class declares, so a trainer
    can carry extra cursor without every other trainer's saved state -- and the
    train_state.json they publish from asdict(state) -- growing a field that
    means nothing to them."""
    paths = _paths(tmp_path)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    save(
        paths, model, opt, _ExtendedState(rows_trained=9, generation_index=2, settled_epochs=5), {}
    )
    assert resume(paths, model, opt, _CPU, state_cls=_ExtendedState) == _ExtendedState(9, 2, 5)

    # The base cursor keeps exactly its own fields, whatever the file holds.
    assert asdict(resume(paths, model, opt, _CPU)) == {"rows_trained": 9, "generation_index": 2}


def test_a_checkpoint_predating_a_field_keeps_its_default(tmp_path):
    """Resuming a run whose checkpoint was written before the field existed."""
    paths = _paths(tmp_path)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save(paths, model, opt, GenerationalState(rows_trained=4, generation_index=1), {})
    assert resume(paths, model, opt, _CPU, state_cls=_ExtendedState) == _ExtendedState(4, 1, 0)
