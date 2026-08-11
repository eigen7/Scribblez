"""The workload registry: every kind of launchable work, declared once.

Each module in this package declares one workload as a WorkloadSpec (see
base.py for the contract). The registry must stay importable without torch or
the engine: heavy code is referenced by dotted path and imported only when it
runs.
"""

from scribblez.workloads import (
    kill_test,
    match_arms,
    max_move_per_lane,
    move_set_eval,
    position_eval,
)
from scribblez.workloads.base import (
    RoleSpec,
    SchedulerHooks,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve,
)

WORKLOADS = {
    spec.name: spec
    for spec in (
        kill_test.SPEC,
        position_eval.SPEC,
        max_move_per_lane.SPEC,
        move_set_eval.SPEC,
        match_arms.SPEC,
    )
}


def get(name: str) -> WorkloadSpec:
    assert name in WORKLOADS, f"unknown workload '{name}'"
    return WORKLOADS[name]


__all__ = [
    "WORKLOADS",
    "get",
    "resolve",
    "RoleSpec",
    "SchedulerHooks",
    "StatsSpec",
    "WorkerContext",
    "WorkloadSpec",
]
