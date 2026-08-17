"""Tests for evidence-trajectory generation (roadmap item 4): the trajectory
.sobs contract end to end (anchor slot, uniform tail, v2 header fields,
proposer hash), the .mset labeling's forced inclusion of simmed candidates,
and the evidence_trajectories workload's cycle and delivery.

The e2e path runs the real binaries (GPU: both generators build TensorRT
engines), over test_slog_writer games and tiny freshly-exported models --
the same fixture recipe as test_move_set_eval_train.
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.move_set_eval.targets import read_mset
from scribblez.sim_evidence.sobs import (
    SOBS_FLAG_TRAJECTORY,
    SOBS_POS_FLAG_UNIFORM_TAIL,
    read_sobs,
    read_sobs_flags,
    read_sobs_proposer_hash,
)
from scribblez.workloads import evidence_trajectories as ET
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.evidence_trajectories import SPEC, EvidenceTrajectoriesParams

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TRAJECTORY_GENERATOR = _ENGINE_DIR / "evidence_trajectory_generator"
TARGET_GENERATOR = _ENGINE_DIR / "move_set_eval_target_generator"
SLOG_WRITER = _ENGINE_DIR / "test_slog_writer"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")


def test_workload_is_registered_with_a_valid_schema():
    spec = workloads.get("evidence_trajectories")
    assert spec is SPEC
    fields = {f.name: f for f in params_mod.schema(EvidenceTrajectoriesParams)}
    assert fields["proposer_model"].kind == "str"
    assert fields["teacher_model"].kind == "str"
    params = EvidenceTrajectoriesParams(proposer_model="/x/student.onnx", teacher_model="/x/t.onnx")
    env = spec.worker_env("t", params, "generate")
    assert params_mod.from_env(EvidenceTrajectoriesParams, env) == params
    role = spec.role("generate")
    assert role.gpu and role.kinds == ("local",)
    assert set(role.stats.phases) == {"gen_s", "traj_s", "mset_s", "upload_s"}


class _CycleRecorder:
    """Stands in for the three subprocess phases of a cycle: self-play writes
    .slog files, the trajectory tool writes .sobs, the labeling writes .mset;
    records what each was asked to process."""

    def __init__(self, monkeypatch, new_slogs: tuple[str, ...], traj_rc: int = 0):
        self.calls: list[tuple[str, list[str]]] = []
        self.new_slogs, self.traj_rc = new_slogs, traj_rc
        monkeypatch.setattr(ET, "run_games", self._run_games)
        monkeypatch.setattr(ET, "run_trajectory_generator", self._trajectories)
        monkeypatch.setattr(mset_targets, "label_stratified", self._label)

    def _run_games(self, out_dir, **kw):
        for stem in self.new_slogs:
            (out_dir / f"{stem}.slog").touch()
        return 0

    def _trajectories(self, pending, params, threads):
        self.calls.append(("traj", [p.stem for p in pending]))
        if self.traj_rc == 0:
            for p in pending:
                p.with_suffix(".sobs").touch()
        return self.traj_rc

    def _label(self, pending, teacher, quotas, positions_per_game, threads, with_sobs=False):
        assert with_sobs, "trajectory pairs must be labeled with their simmed candidates forced"
        assert quotas == mset_targets.StratifiedQuotas(4, 4, 4, 2, 32)
        self.calls.append(("mset", [p.stem for p in pending]))
        for p in pending:
            p.with_suffix(".mset").touch()
        return 0


def test_cycle_sims_only_unsimmed_slogs_and_labels_every_pending_one(tmp_path, monkeypatch):
    """A resumed cycle: `a` was simmed but not labeled when the last run died,
    `b` is fresh. Only `b` is simmed; both are labeled; the phases report."""
    (tmp_path / "a.slog").touch()
    (tmp_path / "a.sobs").touch()
    rec = _CycleRecorder(monkeypatch, new_slogs=("b",))
    r = ET.run_one_cycle(tmp_path, EvidenceTrajectoriesParams(), threads=2)
    assert r.returncode == 0
    assert rec.calls == [("traj", ["b"]), ("mset", ["a", "b"])]
    assert (tmp_path / "a.mset").exists() and (tmp_path / "b.mset").exists()


def test_a_failed_trajectory_phase_skips_the_labeling(tmp_path, monkeypatch):
    """The labeling force-includes candidates from the .sobs, so it must never
    run over a file whose sims failed."""
    rec = _CycleRecorder(monkeypatch, new_slogs=("b",), traj_rc=3)
    r = ET.run_one_cycle(tmp_path, EvidenceTrajectoriesParams(), threads=2)
    assert r.returncode == 3
    assert rec.calls == [("traj", ["b"])]
    assert not (tmp_path / "b.mset").exists()


def test_generate_refuses_a_missing_model(tmp_path, capsys):
    params = EvidenceTrajectoriesParams(proposer_model=str(tmp_path / "no.onnx"), teacher_model="")
    ctx = SimpleNamespace(params=params)
    assert ET.run_generate(ctx) == 1
    err = capsys.readouterr().err
    assert "proposer_model" in err and "teacher_model" in err


class _RecordingSink:
    """A local-sink stand-in: moves the file and records the delivery order."""

    kind = "fake"

    def __init__(self, dest: Path):
        self.dest = dest
        self.delivered: list[str] = []

    def deliver(self, src: Path, rel: str) -> int:
        n = src.stat().st_size
        target = self.dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, target)
        self.delivered.append(Path(rel).name)
        return n


def test_deliver_pairs_carries_the_sobs_sidecar(tmp_path):
    """A trajectory pair is three files; delivery keys on the .mset (the
    completeness token) and the .sobs rides along, delivered first so a
    watcher never sees a labeled pair whose trajectory is still in flight."""
    work = tmp_path / "work"
    work.mkdir()
    for name in ("a.slog", "a.mset", "a.sobs", "b.slog", "b.mset"):
        (work / name).write_bytes(b"x" * 8)
    (work / "c.slog").write_bytes(b"x")  # incomplete: no .mset yet

    sink = _RecordingSink(tmp_path / "store")
    moved, nbytes, _ = pair_store.deliver_pairs(
        sink, work, "w0", ".mset", "slogs", extra_sidecar_exts=(".sobs",)
    )
    assert moved == 2
    assert nbytes == 5 * 8
    assert sink.delivered == ["a-w0.sobs", "a-w0.mset", "a-w0.slog", "b-w0.mset", "b-w0.slog"]
    assert (work / "c.slog").exists()  # incomplete pairs stay put


# --- the e2e path (GPU) ---


def _write_student_onnx(path: Path, shapes: dict) -> None:
    from scribblez.move_set_eval.model import MoveSetEvalModel
    from scribblez.move_set_eval.onnx_export import export_onnx

    torch.manual_seed(0)
    student = MoveSetEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    ).eval()
    # contingent_features matches the session arm get_input_shapes reported
    # under, so the exported width is the width the generator re-derives.
    export_onnx(
        student,
        path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
        opp_leave_input=False,
        move_encoding_version=1,
    )


def _write_teacher_onnx(path: Path, shapes: dict) -> None:
    from scribblez.position_eval.model import PositionEvalModel
    from scribblez.position_eval.onnx_export import export_onnx

    torch.manual_seed(1)
    teacher = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=3,
    ).eval()
    export_onnx(
        teacher,
        path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
        opp_leave_input=False,
    )


@pytest.fixture(scope="module")
def traj_corpus(tmp_path_factory) -> SimpleNamespace:
    """test_slog_writer games with trajectory .sobs sidecars from a tiny
    student, then .mset labels (with --sobs) from a tiny teacher."""
    for binary in (TRAJECTORY_GENERATOR, TARGET_GENERATOR, SLOG_WRITER):
        if not binary.exists():
            pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from scribblez.ffi import get_input_shapes

    d = tmp_path_factory.mktemp("evidence_traj")
    subprocess.run([str(SLOG_WRITER), str(d), "8", "4"], check=True, capture_output=True)
    shapes = {s.name: s.dims for s in get_input_shapes()}
    _write_student_onnx(d / "student.onnx", shapes)
    _write_teacher_onnx(d / "teacher.onnx", shapes)

    traj = subprocess.run(
        [
            str(TRAJECTORY_GENERATOR),
            f"--slog-dir={d}",
            f"--model={d / 'student.onnx'}",
            "--fast-build",
            "--rollouts=8",
            "--proposals-min=1",
            "--proposals-max=3",
            "--proposal-pool=8",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert traj.returncode == 0, f"trajectory generator failed: {traj.stderr}"

    mset = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={d}",
            f"--model={d / 'teacher.onnx'}",
            "--fast-build",
            "--sobs",
            "--positions-per-game=4",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert mset.returncode == 0, f"target generator failed: {mset.stderr}"
    return SimpleNamespace(dir=d)


def test_trajectory_sobs_contract(traj_corpus):
    """The .sobs v2 trajectory contract: flags, proposer hash, per-position
    legal counts, the anchor's raw-score supremacy, the uniform tail, and
    trajectory sizes within the configured bounds."""
    sobs_files = sorted(traj_corpus.dir.glob("*.sobs"))
    assert sobs_files
    total_positions = 0
    for f in sobs_files:
        assert read_sobs_flags(f) & SOBS_FLAG_TRAJECTORY
        assert len(read_sobs_proposer_hash(f)) == 16  # nn::content_hash's width
        for pos in read_sobs(f):
            total_positions += 1
            k = len(pos.moves)
            assert 0 < k <= pos.num_legal_moves
            # anchor + [1..3] proposals + at most one uniform tail.
            assert k <= 1 + 3 + 1
            # The anchor holds the position's highest raw score, so no other
            # simmed candidate may beat it.
            assert all(int(m["score"]) <= int(pos.moves[0]["score"]) for m in pos.moves[1:])
            if pos.has_uniform_tail:
                assert pos.flags & SOBS_POS_FLAG_UNIFORM_TAIL
                assert max(pos.evidence_prefix_sizes()) == k - 1
            else:
                # No tail only when the trajectory exhausted the legal set.
                assert k == pos.num_legal_moves
                assert max(pos.evidence_prefix_sizes()) == k
            # Rollout counts match the header (every candidate was simmed).
            assert all(int(o["n"]) == pos.rollouts == 8 for o in pos.obs)
    assert total_positions > 0


def test_mset_labeling_covers_the_simmed_candidates(traj_corpus):
    """The roadmap item-4 invariant: every trajectory candidate appears among
    its position's value-labeled .mset candidates."""
    for sobs in sorted(traj_corpus.dir.glob("*.sobs")):
        mset = read_mset(sobs.with_suffix(".mset"))
        labeled = {(p.game_index, p.turn_index): p.moves.tobytes() for p in mset.positions}
        for pos in read_sobs(sobs):
            key = (pos.game_index, pos.turn_index)
            assert key in labeled, f"simmed position {key} was not labeled"
            move_bytes = labeled[key]
            record = np.frombuffer(move_bytes, dtype=pos.moves.dtype)
            stored = {m.tobytes() for m in record}
            for m in pos.moves:
                assert m.tobytes() in stored, f"simmed candidate missing at {key}"


def test_sobs_positions_outside_the_sample_fail_loudly(traj_corpus, tmp_path):
    """The subset guard: labeling a trajectory pair with a smaller
    --positions-per-game than the trajectory run's leaves simmed positions
    outside the labeled sample, which must fail rather than silently waste
    the corpus's most expensive rows."""
    src = sorted(traj_corpus.dir.glob("*.slog"))[0]
    shutil.copy(src, tmp_path / src.name)
    shutil.copy(src.with_suffix(".sobs"), (tmp_path / src.name).with_suffix(".sobs"))
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-file={tmp_path / src.name}",
            f"--model={traj_corpus.dir / 'teacher.onnx'}",
            "--fast-build",
            "--sobs",
            "--positions-per-game=1",  # the trajectory run sampled 2 per game
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "is not in this run's sample" in result.stderr


def test_sobs_flag_requires_the_sidecar(traj_corpus, tmp_path):
    """--sobs without a .sobs beside the .slog must fail loudly, not label a
    trajectory pair without its forced candidates."""
    src = sorted(traj_corpus.dir.glob("*.slog"))[0]
    shutil.copy(src, tmp_path / src.name)
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-file={tmp_path / src.name}",
            f"--model={traj_corpus.dir / 'teacher.onnx'}",
            "--fast-build",
            "--sobs",
            "--positions-per-game=4",
            "--threads=2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no .sobs sidecar" in result.stderr


def test_sobs_flag_rejects_full_sweep():
    if not TARGET_GENERATOR.exists():
        pytest.skip("engine binaries not built")
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            "--slog-dir=/nonexistent",
            "--model=/nonexistent/model.onnx",  # parse-time requirement; never loaded
            "--sobs",
            "--full-sweep",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert result.returncode != 0
    assert "--sobs applies to the stratified selection" in result.stderr
