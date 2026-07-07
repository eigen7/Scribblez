"""The workload registry: every kind of launchable work, declared once.

A WorkloadSpec ties together a workload's name, its parameter dataclass (see
scribblez/params.py -- the single source of truth from which CLI flags, worker
env vars, the dashboard's config form, and validation all derive), where its
per-tag data lives, and its scheduling traits. Consumers:

  - the master dashboard (task creation, worker launch, tag enumeration)
  - the cloud fleet CLI (scripts/cloud_fleet.py)
  - the worker entrypoint (py/cloud/worker_entrypoint.py), which reads
    SCZ_WORKLOAD and reconstructs the params from the env
"""

from dataclasses import dataclass
from pathlib import Path

from scribblez import params as params_mod
from scribblez.kill_test_gen import DATA_ROOT as KILL_TEST_DATA_ROOT
from scribblez.kill_test_gen import KillTestParams


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str  # human-readable, shown in the dashboard's workload picker
    params_cls: type
    data_root: Path  # per-tag data lives at data_root/<tag>
    # Whether cloud pods for this workload are rented interruptible (spot):
    # cheaper, but Runpod may stop them at any time. Only for workloads that
    # tolerate preemption (the reconcile loop restarts reclaimed pods).
    interruptible: bool

    def data_dir(self, tag: str) -> Path:
        return self.data_root / tag

    def worker_env(self, tag: str, params) -> dict[str, str]:
        """The SCZ_* environment defining this work for a worker entrypoint
        (local or cloud); worker-level knobs (sink, threads, worker id) and R2
        credentials are layered on top by the launcher."""
        return {"SCZ_WORKLOAD": self.name, "SCZ_TAG": tag, **params_mod.to_env(params)}


WORKLOADS = {
    "kill_test": WorkloadSpec(
        name="kill_test",
        title="Generate kill-test data",
        params_cls=KillTestParams,
        data_root=KILL_TEST_DATA_ROOT,
        interruptible=True,
    ),
}


def get(name: str) -> WorkloadSpec:
    assert name in WORKLOADS, f"unknown workload '{name}'"
    return WORKLOADS[name]
