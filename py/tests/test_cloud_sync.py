"""What the per-task sync pulls (scripts/cloud_sync.py), with and without a
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
        workloads.WorkloadSpec, "paths", lambda self, tag: TagPaths(tag, self.name, tmp_path)
    )
    return workloads.get("position_eval")


def _pulled(rc):
    return [(a[0], *a[1:]) for a in rc.calls]


def test_a_generator_only_tag_pulls_staging_stats_and_params(spec, monkeypatch):
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    assert cloud_sync.sync_once(R2, spec, "t") == 0
    root = spec.paths("t").root
    assert _pulled(rc) == [
        ("copy", "r2:b/position_eval/t/staging", str(root / "data" / "staging")),
        ("copy", "r2:b/position_eval/t/stats", str(root / "stats")),
        ("copy", "r2:b/position_eval/t/params", str(root / "params")),
    ]


def test_trainer_outputs_are_pulled_immutable_ones_by_size(spec, monkeypatch):
    """Records and exports never change once written, so they are compared
    by size (an S3 listing carries no modtime; the default would HEAD every
    export each pass); the checkpoint is rewritten in place and is not. The
    cursor file is pulled only once the bucket has it."""
    rc = _Rclone()
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    assert cloud_sync.sync_once(R2, spec, "t", trainer_outputs=True) == 0
    root = spec.paths("t").root
    assert _pulled(rc)[3:] == [
        ("copy", "--size-only", "r2:b/position_eval/t/records", str(root / "records")),
        ("copy", "--size-only", "r2:b/position_eval/t/models", str(root / "models")),
        ("copy", "r2:b/position_eval/t/checkpoints", str(root / "checkpoints")),
        ("lsf", "r2:b/position_eval/t/train_state.json"),
    ]
    rc = _Rclone(present={"train_state.json"})
    monkeypatch.setattr(cloud_sync, "rclone", rc)
    assert cloud_sync.sync_once(R2, spec, "t", trainer_outputs=True) == 0
    assert _pulled(rc)[-1] == (
        "copyto", "r2:b/position_eval/t/train_state.json", str(root / "train_state.json")
    )  # fmt: skip
