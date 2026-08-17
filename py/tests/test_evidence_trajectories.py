"""Tests for evidence-trajectory generation (roadmap item 4): the trajectory
.sobs contract end to end (anchor slot, uniform tail, v2 header fields,
proposer hash), the .mset labeling's forced inclusion of simmed candidates,
and the workload-side pair selection and delivery.

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
from scribblez.move_set_eval.targets import read_mset
from scribblez.sim_evidence.sobs import (
    SOBS_FLAG_TRAJECTORY,
    SOBS_POS_FLAG_UNIFORM_TAIL,
    read_sobs,
    read_sobs_flags,
    read_sobs_proposer_hash,
)
from scribblez.workloads import pair_store
from scribblez.workloads.move_set_eval import MoveSetEvalParams, sweep_pair, traj_pair

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TRAJECTORY_GENERATOR = _ENGINE_DIR / "evidence_trajectory_generator"
TARGET_GENERATOR = _ENGINE_DIR / "move_set_eval_target_generator"
SLOG_WRITER = _ENGINE_DIR / "test_slog_writer"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")


def _params(**kw) -> MoveSetEvalParams:
    return MoveSetEvalParams(**kw)


def test_traj_pairs_are_stem_stable_and_never_sweep_pairs():
    params = _params(traj_every=3, sweep_every=4)
    stems = [f"batch-{i:04d}" for i in range(200)]
    chosen = [s for s in stems if traj_pair(s, params)]
    assert chosen  # the hash actually selects some
    assert len(chosen) < len(stems)
    for s in chosen:
        assert not sweep_pair(s, params.sweep_every)
        assert traj_pair(s, params)  # decision is a pure function of the stem
    # Disabled entirely at traj_every=0.
    assert not [s for s in stems if traj_pair(s, _params(traj_every=0))]


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


# --- the .gcg front-end (position sets) ---

POSITION_SET = (
    Path(__file__).resolve().parents[2] / "positions" / "NWL23" / "position-eval-test-dataset"
)


def test_gcg_mode_writes_one_sidecar_per_position(traj_corpus, tmp_path):
    """A .gcg position set gets one trajectory .sobs per file into --out-dir:
    the single position keyed (0, decision turn), the same trajectory contract
    as the .slog path, and existing outputs skipped on a rerun."""
    gcgs = [POSITION_SET / "pos-1.gcg", POSITION_SET / "pos-2.gcg"]
    out = tmp_path / "sobs"
    cmd = [
        str(TRAJECTORY_GENERATOR),
        *[f"--gcg={g}" for g in gcgs],
        f"--out-dir={out}",
        f"--model={traj_corpus.dir / 'student.onnx'}",
        "--fast-build",
        "--rollouts=8",
        "--proposals-min=1",
        "--proposals-max=3",
        "--proposal-pool=8",
        "--threads=2",
        "--seed=7",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"gcg mode failed: {r.stderr}"
    for gcg in gcgs:
        sobs = out / f"{gcg.stem}.sobs"
        assert read_sobs_flags(sobs) & SOBS_FLAG_TRAJECTORY
        positions = read_sobs(sobs)
        assert len(positions) == 1
        pos = positions[0]
        n_turns = sum(1 for line in gcg.read_text().splitlines() if line.startswith(">"))
        assert (pos.game_index, pos.turn_index) == (0, n_turns - 1)
        assert 0 < len(pos.moves) <= 1 + 3 + 1
        assert all(int(m["score"]) <= int(pos.moves[0]["score"]) for m in pos.moves[1:])
        assert all(int(o["n"]) == pos.rollouts == 8 for o in pos.obs)
    # A rerun sims nothing (the outputs exist), and a mixed invocation refuses.
    before = {p: p.stat().st_mtime_ns for p in out.glob("*.sobs")}
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "0 gcg" not in r.stderr  # nothing pending: silent
    assert {p: p.stat().st_mtime_ns for p in out.glob("*.sobs")} == before
    r = subprocess.run(cmd + [f"--slog-dir={traj_corpus.dir}"], capture_output=True, text=True)
    assert r.returncode != 0 and "not both" in r.stderr


def test_position_set_cache_regenerates_only_stale_or_missing(tmp_path, monkeypatch):
    """ensure_sobs keys the cache on (proposer bytes, recipe), asks the
    generator only for positions whose sidecar is absent or whose .gcg changed,
    and records what each sidecar was generated from."""
    from scribblez.sim_evidence import position_sets as PS

    set_dir = tmp_path / "myset"
    set_dir.mkdir()
    for name in ("a", "b"):
        (set_dir / f"{name}.gcg").write_text(f"#character-encoding UTF-8\n>P1: {name}\n")
    model = tmp_path / "student.onnx"
    model.write_bytes(b"model-v1")
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        out = Path(next(c for c in cmd if c.startswith("--out-dir=")).split("=", 1)[1])
        for c in cmd:
            if c.startswith("--gcg="):
                (out / (Path(c.split("=", 1)[1]).stem + ".sobs")).write_bytes(b"sobs")

    monkeypatch.setattr(PS.subprocess, "run", fake_run)
    recipe = PS.TrajectoryRecipe(rollouts=16, open_leaves=True)
    got = PS.ensure_sobs(set_dir, model, recipe, threads=2, mount_root=tmp_path / "mount")
    assert set(got) == {"a", "b"} and all(p.exists() for p in got.values())
    assert len(calls) == 1
    assert "--open-leaves" in calls[0] and "--rollouts=16" in calls[0]
    assert sum(c.startswith("--gcg=") for c in calls[0]) == 2

    # Nothing changed: no generator run.
    PS.ensure_sobs(set_dir, model, recipe, threads=2, mount_root=tmp_path / "mount")
    assert len(calls) == 1
    # One .gcg edited: only it is regenerated.
    (set_dir / "b.gcg").write_text("#character-encoding UTF-8\n>P1: b2\n")
    PS.ensure_sobs(set_dir, model, recipe, threads=2, mount_root=tmp_path / "mount")
    assert len(calls) == 2
    assert [c for c in calls[1] if c.startswith("--gcg=")] == [f"--gcg={set_dir / 'b.gcg'}"]
    # A different proposer or recipe is a different cache directory.
    model.write_bytes(b"model-v2")
    d1 = PS.cache_dir(set_dir, model, recipe, tmp_path / "mount")
    d2 = PS.cache_dir(set_dir, model, PS.TrajectoryRecipe(rollouts=32), tmp_path / "mount")
    assert d1 != d2 and d1 != got["a"].parent
