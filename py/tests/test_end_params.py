"""End parameters (scribblez/params.py): a budget at which a tag's work
finishes on its own, UNBOUNDED (-1) for "run until paused", 0 accepted as its
legacy alias.

The danger this pins is a comparison that reads -1 as a budget already spent:
the worker would exit 0 at once and look finished, so a long run would silently
become an empty one. Every workload's run-until-end predicate is driven here
with -1 and with 0, and must keep going."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.params import UNBOUNDED, param, reached, unbounded

UNBOUNDED_SPELLINGS = [UNBOUNDED, 0]


def test_unbounded_and_reached():
    assert unbounded(-1) and unbounded(0) and not unbounded(1)
    assert reached(5, 5) and reached(6, 5) and not reached(4, 5)
    for limit in UNBOUNDED_SPELLINGS:
        assert not reached(0, limit) and not reached(10**12, limit)


@dataclass(frozen=True)
class _Budgeted:
    rows: int = param(-1, "budget", end=True)


def test_validation_accepts_unbounded_and_budgets_and_refuses_below():
    for value in (-1, 0, 1, 10**9):
        assert params_mod.validate(_Budgeted, {"rows": value}).rows == value
    with pytest.raises(params_mod.ParamsError, match="rows"):
        params_mod.validate(_Budgeted, {"rows": -2})


def test_the_public_schema_marks_end_parameters():
    (field,) = params_mod.public_schema(_Budgeted)
    assert field["end"] is True


def test_an_end_parameter_must_be_an_int():
    @dataclass(frozen=True)
    class Bad:
        frac: float = param(0.5, "not a count", end=True)

    with pytest.raises(AssertionError, match="int"):
        params_mod.schema(Bad)


# The end parameters each workload declares. Adding or removing one should be a
# deliberate edit here, since the queue's "has an end condition" reads them.
END_PARAMS = {
    "position_eval": {"max_rows"},
    "max_move_per_lane": {"max_rows"},
    "move_set_eval": {"target_pairs", "train_epochs"},
    "evidence_trajectories": {"target_pairs", "train_epochs"},
    "blind_spots": {"target_positions"},
    "kill_test": set(),
    "match_arms": set(),
    "transfer_test": {"target_positions"},
    "transfer_reader": {"train_steps"},
}


def test_every_workload_declares_its_end_parameters():
    assert set(END_PARAMS) == set(workloads.WORKLOADS)
    for name, spec in workloads.WORKLOADS.items():
        declared = {f.name for f in params_mod.schema(spec.params_cls) if f.end}
        assert declared == END_PARAMS[name], name


@pytest.mark.parametrize("limit", UNBOUNDED_SPELLINGS)
def test_generational_trainers_keep_training_when_unbounded(limit):
    from scribblez.generational.checkpoint import GenerationalState
    from scribblez.max_move_per_lane import trainer as mmpl
    from scribblez.position_eval import trainer as pe

    state = GenerationalState(rows_trained=10**9)
    for rows_left in (pe._rows_left, mmpl._rows_left):
        assert rows_left(SimpleNamespace(max_rows=limit), state)
        assert not rows_left(SimpleNamespace(max_rows=100), state)


@pytest.mark.parametrize("limit", UNBOUNDED_SPELLINGS)
def test_pair_trainers_keep_training_when_unbounded(limit):
    from scribblez.evidence import trainer as evidence
    from scribblez.move_set_eval import trainer as mset

    for module in (mset, evidence):
        state = SimpleNamespace(settled_epochs=10**6)
        assert module.epochs_left(SimpleNamespace(train_epochs=limit), state)
        assert not module.epochs_left(SimpleNamespace(train_epochs=20), state)


@pytest.mark.parametrize("limit", UNBOUNDED_SPELLINGS)
def test_the_scheduler_keeps_the_generators_when_unbounded(tmp_path, limit):
    import json

    from scribblez.generational import scheduler
    from scribblez.paths import POSITION_EVAL, TagPaths

    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    paths.train_state_path.parent.mkdir(parents=True)
    paths.train_state_path.write_text(json.dumps({"rows_trained": 10**9}))
    assert not scheduler.complete(None, paths, SimpleNamespace(max_rows=limit))
    assert scheduler.complete(None, paths, SimpleNamespace(max_rows=100))
