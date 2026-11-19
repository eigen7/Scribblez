"""The known positions as a role's inputs (workloads/known_position_inputs.py):
a local slot reads the repo's copies, a remote one, which has no repo copy,
the ones the controller staged under its tag."""

import shutil
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.workloads import known_position_inputs
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.leaf_eval_truth import ROLE, SPEC, LeafEvalTruthParams


def _ctx(mount: Path, kind: str) -> WorkerContext:
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role(ROLE),
        tag="t",
        params=LeafEvalTruthParams(),
        worker_id="w0",
        threads=1,
        max_cycles=0,
        sink=None,
        mount_root=mount,
        kind=kind,
    )


def test_a_local_slot_reads_the_repos_positions(tmp_path):
    got = known_position_inputs.load(_ctx(tmp_path, "local"))
    assert got == known_positions.load()


def test_a_remote_slot_reads_the_staged_positions(tmp_path, monkeypatch):
    staged = SPEC.paths("t", tmp_path).root / known_position_inputs.INPUT_DIR
    staged.mkdir(parents=True)
    for f in known_positions.files():
        shutil.copy(f, staged / f.name)
    monkeypatch.setattr(known_positions, "DIRECTORY", tmp_path / "no-repo")
    got = known_position_inputs.load(_ctx(tmp_path, "ssh"))
    assert [p.id for p in got] == [p.id for p in known_positions.load(staged)]
    assert all(p.gcg.parent == staged and p.gcg.is_file() for p in got)
