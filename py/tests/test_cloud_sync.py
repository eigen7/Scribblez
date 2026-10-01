"""What the per-task sync pulls (py/scripts/cloud_sync.py), with and without a
trainer delivering through the bucket."""

import sys
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


def test_a_generator_only_tag_pulls_staging_stats_and_params(spec, monkeypatch):
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    assert cloud_sync.sync_once(R2, spec, spec.paths("t")) == 0
    root = spec.paths("t").root
    assert _pulled(rc) == [
        ("copy", "r2:b/position_eval/t/staging", str(root / "data" / "staging")),
        ("copy", "r2:b/position_eval/t/stats", str(root / "stats")),
        ("copy", "r2:b/position_eval/t/params", str(root / "params")),
    ]


def test_without_data_a_data_homes_staging_is_left_in_the_bucket(spec, monkeypatch):
    """A data home moves bucket staging chunks in itself; a pull here would
    bring back chunks it already assigned. The records still come down."""
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    monkeypatch.setattr(cloud_sync, "load_credentials", lambda: SimpleNamespace(r2=R2))
    monkeypatch.setattr(
        sys, "argv", ["cloud_sync", "--workload", "position_eval", "-t", "t", "--no-data"]
    )
    assert cloud_sync.main() == 0
    root = spec.paths("t").root
    assert _pulled(rc) == [
        ("copy", "r2:b/position_eval/t/stats", str(root / "stats")),
        ("copy", "r2:b/position_eval/t/params", str(root / "params")),
    ]


def test_trainer_outputs_are_pulled_immutable_ones_by_size(spec, monkeypatch):
    """Records and exports never change once written, so they are compared by
    size alone: an S3 listing carries no modtime, and rclone's default check
    would HEAD every export on every pass. The checkpoint and cursor come as a
    state pair under the cursor rule (state_pair.restore), never as files a
    stale bucket could overwrite fresher local state with."""
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    restored = []
    monkeypatch.setattr(
        cloud_sync.state_pair, "restore", lambda paths, sink: restored.append((paths.tag, sink))
    )
    assert cloud_sync.sync_once(R2, spec, spec.paths("t"), trainer_outputs=True) == 0
    root = spec.paths("t").root
    assert _pulled(rc)[3:] == [
        ("copy", "--size-only", "r2:b/position_eval/t/records", str(root / "records")),
        ("sync", "--size-only", "r2:b/position_eval/t/models", str(root / "models")),
        ("lsf", "r2:b/position_eval/t/scheduler_state.json"),  # a data home's heartbeat
    ]
    [(tag, sink)] = restored
    assert tag == "t" and isinstance(sink, cloud_sync.R2Sink)
