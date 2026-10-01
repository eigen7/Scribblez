"""What the per-task sync pulls (py/scripts/cloud_sync.py), with and without a
trainer delivering through the bucket."""

from types import SimpleNamespace

import pytest
from scribblez import workloads
from scribblez.paths import TagPaths
from scripts import cloud_sync

R2 = SimpleNamespace(bucket="b")


class _Rclone:
    def __init__(self, present=()):
        self.calls = []
        self.present = set(present)  # objects lsf finds

    def __call__(self, r2, *args, capture=False, input_text=None):
        self.calls.append(args)
        if args[0] == "lsf":
            name = args[1].rsplit("/", 1)[-1]
            return SimpleNamespace(returncode=0, stdout=name + "\n" if name in self.present else "")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def spec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        workloads.WorkloadSpec,
        "paths",
        lambda self, tag, mount_root=None: TagPaths(tag, self.name, tmp_path),
    )
    return workloads.get("position_eval")


def _pulled(rc):
    return [(a[0], *a[1:]) for a in rc.calls]


def test_only_the_data_rented_generators_deliver_is_pulled(spec, monkeypatch):
    """Records and a trainer's outputs reach the controller over ssh
    (WorkerManager._transfer_target); the bucket carries only generator data."""
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    assert cloud_sync.sync_once(R2, spec, spec.paths("t")) == 0
    root = spec.paths("t").root
    assert _pulled(rc) == [("copy", "r2:b/position_eval/t/staging", str(root / "data" / "staging"))]


def test_a_failed_pull_fails_the_pass(spec, monkeypatch):
    monkeypatch.setattr(
        cloud_sync,
        "rclone",
        lambda r2, *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="x"),
    )
    assert cloud_sync.sync_once(R2, spec, spec.paths("t")) == 1
