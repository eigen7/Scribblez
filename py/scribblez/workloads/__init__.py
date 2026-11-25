"""The workload registry: every kind of launchable work, declared once.

Each module in this package declares one workload as a WorkloadSpec (see
base.py for the contract). The registry must stay importable without torch or
the engine: heavy code is referenced by dotted path and imported only when it
runs.
"""

from scribblez.workloads import (
    blind_spots,
    deep_bot_leaf_eval,
    leaf_bot_generation,
    leaf_eval_hasty_rollouts,
    leave_inference,
    match_arms,
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
        position_eval.SPEC,
        match_arms.SPEC,
        blind_spots.SPEC,
        leave_inference.SPEC,
        deep_bot_leaf_eval.SPEC,
        leaf_eval_hasty_rollouts.SPEC,
        leaf_bot_generation.SPEC,
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
