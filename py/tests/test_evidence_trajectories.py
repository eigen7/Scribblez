"""Tests for evidence-trajectory generation (roadmap item 4): the trajectory
.sobs contract end to end (anchor slot, per-record roles, v4 header fields,
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
from scribblez.evidence.dataset import assemble_subset, gain_targets
from scribblez.move_set_eval.targets import read_mset
from scribblez.sim_evidence.sobs import (
    MOVE_DTYPE,
    RECORD_DTYPE,
    ROLE_ANCHOR,
    ROLE_OFF_POLICY,
    ROLE_ON_POLICY,
    SOBS_FLAG_TRAJECTORY,
    SobsPosition,
    read_sobs,
    read_sobs_flags,
    read_sobs_proposer_hash,
)
from scribblez.workloads import evidence_trajectories as ET
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.evidence_trajectories import (
    SPEC,
    EvidenceTrajectoriesParams,
    max_evidence_width,
    max_off_policy,
    max_pool_width,
)

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


def _synthetic_position(roles: np.ndarray, num_legal_moves: int = 64) -> SobsPosition:
    k = len(roles)
    return SobsPosition(
        game_index=0,
        turn_index=0,
        rollouts=100,
        base_seed=0,
        num_legal_moves=num_legal_moves,
        flags=0,
        moves=np.zeros(k, MOVE_DTYPE),
        obs=np.zeros(k, RECORD_DTYPE["obs"]),
        roles=roles,
    )


def test_v4_roles_split_evidence_from_labels_only_draws():
    """A synthetic v4 position: evidence prefixes cover only the anchor and the
    on-policy records; off-policy draws sit outside every prefix and never enter
    the gain baseline (the split item 5's subset assembly depends on)."""
    roles = np.array(
        [ROLE_ANCHOR, ROLE_ON_POLICY, ROLE_ON_POLICY, ROLE_OFF_POLICY, ROLE_OFF_POLICY],
        dtype=np.uint8,
    )
    pos = _synthetic_position(roles)
    assert pos.num_evidence == 3
    assert list(pos.evidence_prefix_sizes()) == [0, 1, 2, 3]
    assert pos.evidence_mask.tolist() == [True, True, True, False, False]
    # The off-policy draw carries the largest value, but because it is never in
    # a prefix it never raises the best-so-far baseline the gain is measured
    # against -- so no candidate's gain is suppressed by a labels-only sim.
    value = np.array([0.2, 0.5, 0.3, 0.9, 0.4], dtype=np.float32)
    for prefix in pos.evidence_prefix_sizes():
        subset = np.zeros(len(value), dtype=bool)
        subset[:prefix] = True  # a leading prefix is one valid subset
        best = float(value[subset].max()) if prefix else 0.0
        assert best <= 0.5  # the off-policy 0.9 is never in an evidence subset
        np.testing.assert_allclose(gain_targets(value, subset), np.maximum(value - best, 0.0))


def test_v4_num_evidence_is_order_robust():
    """Should an off-policy record ever precede an on-policy one, the evidence
    prefix stops at it rather than admitting a labels-only record."""
    roles = np.array([ROLE_ANCHOR, ROLE_OFF_POLICY, ROLE_ON_POLICY], dtype=np.uint8)
    pos = _synthetic_position(roles, num_legal_moves=10)
    assert pos.num_evidence == 1
    assert list(pos.evidence_prefix_sizes()) == [0, 1]


def test_assemble_subset_never_draws_a_labels_only_record():
    """Subset assembly reads eligibility from the role, not the storage order:
    an off-policy draw is never a member (its held-out row is what carries the
    floor label), and the num_evidence clamp keeps an on-policy record sitting
    behind an off-policy one out of every subset too."""
    # off-policy at slot 1, an on-policy behind it -> num_evidence clamps to 1.
    roles = np.array([ROLE_ANCHOR, ROLE_OFF_POLICY, ROLE_ON_POLICY], dtype=np.uint8)
    pos = _synthetic_position(roles, num_legal_moves=10)
    assert pos.num_evidence == 1
    rng = np.random.default_rng(0)
    sizes = set()
    for _ in range(200):
        mask = assemble_subset(rng, pos)
        sizes.add(int(mask.sum()))
        assert not mask[1] and not mask[2]  # neither the off-policy nor the clamped on-policy
    assert sizes == {0, 1}  # empty or the anchor alone


def test_trajectory_width_formulas():
    """The two guard widths: max_evidence_width is the padded evidence capacity
    (anchor + on-policy), max_pool_width the full record count the train-role
    guard checks a corpus against (adds the off-policy floor). Pinned so a
    swapped or mis-summed formula is caught without the GPU e2e path."""
    params = EvidenceTrajectoriesParams()
    assert max_evidence_width(params) == 1 + params.on_policy_max
    assert max_off_policy(params) == params.off_policy_count
    assert max_pool_width(params) == 1 + params.on_policy_max + max_off_policy(params)
    # The pool the guard admits is strictly wider than the padded evidence input.
    assert max_pool_width(params) > max_evidence_width(params)


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
    # The export takes the widths get_input_shapes reported, so the exported
    # width is the width the generator re-derives.
    export_onnx(
        student,
        path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
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
            "--on-policy-min=1",
            "--on-policy-max=3",
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
    """The .sobs v4 trajectory contract: flags, proposer hash, per-position
    legal counts, the anchor's raw-score supremacy, and the per-record roles
    (anchor, then on-policy, then off-policy)."""
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
            assert pos.flags == 0  # no position flags at v4
            roles = np.asarray(pos.roles)
            # Storage order: exactly one anchor first, then on-policy, then the
            # off-policy floor -- so evidence-eligible records form a prefix.
            assert roles[0] == ROLE_ANCHOR
            assert (roles == ROLE_ANCHOR).sum() == 1
            evidence = roles != ROLE_OFF_POLICY
            num_evidence = pos.num_evidence
            assert evidence[:num_evidence].all() and not evidence[num_evidence:].any()
            assert set(roles[1:num_evidence].tolist()) <= {ROLE_ON_POLICY}
            # Off-policy draws are labels-only: the evidence prefix stops at them.
            assert max(pos.evidence_prefix_sizes()) == num_evidence
            # The anchor holds the position's highest raw score, so no other
            # simmed candidate may beat it.
            assert all(int(m["score"]) <= int(pos.moves[0]["score"]) for m in pos.moves[1:])
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

# Frozen .gcg fixtures. Live position sets under positions/ are regenerated and
# renamed; a test must never read one.
TEST_DATA = Path(__file__).resolve().parents[2] / "engine" / "tests" / "data"


def _as_position(src: Path, dst: Path) -> int:
    """A position-set .gcg from a position-eval one: the last recorded move
    is dropped and its rack becomes the mover's #RackN pragma, so the file's
    final state is that decision. Returns the recorded move count."""
    lines = src.read_text().splitlines()
    events = [i for i, line in enumerate(lines) if line.startswith(">")]
    last = lines[events[-1]]
    name, rack = last[1:].split(":", 1)[0], last.split()[1]
    seat = 1 if name.endswith("_1") else 2
    lines = lines[: events[-1]]
    lines.insert(events[0], f"#Rack{seat} {rack}")
    dst.write_text("\n".join(lines) + "\n")
    return len(events) - 1


def test_gcg_mode_writes_one_sidecar_per_position(traj_corpus, tmp_path):
    """A .gcg position set gets one trajectory .sobs per file into --out-dir:
    the single position keyed (0, decision turn), the same trajectory contract
    as the .slog path, and existing outputs skipped on a rerun."""
    set_dir = tmp_path / "set"
    set_dir.mkdir()
    names = ("ole", "violets")
    turns = {n: _as_position(TEST_DATA / f"{n}.gcg", set_dir / f"{n}.gcg") for n in names}
    out = tmp_path / "sobs"
    cmd = [
        str(TRAJECTORY_GENERATOR),
        f"--gcg-dir={set_dir}",
        f"--out-dir={out}",
        f"--model={traj_corpus.dir / 'student.onnx'}",
        "--fast-build",
        "--rollouts=8",
        "--on-policy-min=1",
        "--on-policy-max=3",
        "--threads=2",
        "--seed=7",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"gcg mode failed: {r.stderr}"
    for name, n_turns in turns.items():
        sobs = out / f"{name}.sobs"
        assert read_sobs_flags(sobs) & SOBS_FLAG_TRAJECTORY
        positions = read_sobs(sobs)
        assert len(positions) == 1
        pos = positions[0]
        assert (pos.game_index, pos.turn_index) == (0, n_turns)
        # anchor + [1..3] on-policy + the default off-policy floor (3 uniform draws).
        assert 0 < len(pos.moves) <= 1 + 3 + 3
        assert all(int(m["score"]) <= int(pos.moves[0]["score"]) for m in pos.moves[1:])
        assert all(int(o["n"]) == pos.rollouts == 8 for o in pos.obs)
    # A rerun sims nothing (the outputs exist), and a mixed invocation refuses.
    before = {p: p.stat().st_mtime_ns for p in out.glob("*.sobs")}
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0
    assert {p: p.stat().st_mtime_ns for p in out.glob("*.sobs")} == before
    r = subprocess.run(cmd + [f"--slog-dir={traj_corpus.dir}"], capture_output=True, text=True)
    assert r.returncode != 0 and "not both" in r.stderr
    # A file without the mover's rack pragma is refused by name, before any sim.
    (set_dir / "norack.gcg").write_text((TEST_DATA / "boreal.gcg").read_text())
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "norack.gcg" in r.stderr and "#Rack" in r.stderr


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


# --- the trajectory pane's model side (scribblez.evidence.trajectory_view) ---

KWG = Path("/workspace/mount/lexica/NWL23.kwg")


def test_gcg_position_inputs_reads_the_exhibit_decision():
    """The FFI's reading of the exhibit position: the mover's arm-sized row,
    the pre-move differential, the full equity-ranked legal move list, and a
    board bundle whose notations line up with it (E11 GAVE plays through the
    A of INCASED, "E11 G.VE")."""
    if not (KWG.exists() and LEAVES.exists()):
        pytest.skip("NWL23 lexicon / leaves not installed")
    from scribblez.ffi import gcg_position_board_json, gcg_position_inputs

    text = (TEST_DATA / "egotize-lane.gcg").read_text()
    inputs = gcg_position_inputs(text, opp_leave_input=True, spatial_planes=87, scalar_size=163)
    assert inputs.input_spatial.shape == (87, 15, 15) and inputs.input_scalar.shape == (163,)
    assert inputs.score_diff == 440 - 387
    bundle = gcg_position_board_json(text, open_leaves=True)
    assert len(bundle["moves"]) == len(inputs.moves) > 100
    assert "E11 G.VE" in bundle["moves"]
    assert bundle["mover"] == 0 and bundle["scores"] == [440, 387]
    assert [t["letter"] for t in bundle["rack"]] == list("AEEGSTV")
    # A width the arm does not encode is refused, naming the mismatch: the
    # hidden-leaves arm has no opponent-leave block, so it is 27 scalars short.
    with pytest.raises(ValueError, match="floats"):
        gcg_position_inputs(text, opp_leave_input=False, spatial_planes=85, scalar_size=163)


def _tiny_evidence_checkpoint(trained: bool):
    """A tiny frozen-backbone model under the session's default arm (the
    fixture student's), as either checkpoint kind."""
    from scribblez.evidence.checkpoints import EvidenceCheckpoint
    from scribblez.ffi import get_input_shapes
    from scribblez.move_set_eval.model import MoveSetEvalModel

    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(3)
    model = MoveSetEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    if trained:  # give the fusion stage non-trivial weights, as training would
        with torch.no_grad():
            for p in model.evidence_parameters():
                p.add_(torch.randn_like(p) * 0.1)
    model.freeze_backbone()
    cfg = {
        "spatial_planes": shapes["input_spatial"][0],
        "scalar_size": shapes["input_scalar"][0],
        "open_leaves": False,
    }
    return EvidenceCheckpoint(model.eval(), cfg, trained=trained)


@pytest.fixture(scope="module")
def gcg_set(traj_corpus, tmp_path_factory) -> SimpleNamespace:
    """A two-position .gcg set with trajectory sidecars from the fixture
    student (gcg mode, 8 rollouts)."""
    set_dir = tmp_path_factory.mktemp("gcgset")
    for name in ("ole", "violets"):
        _as_position(TEST_DATA / f"{name}.gcg", set_dir / f"{name}.gcg")
    out = tmp_path_factory.mktemp("gcgset_sobs")
    r = subprocess.run(
        [
            str(TRAJECTORY_GENERATOR),
            f"--gcg-dir={set_dir}",
            f"--out-dir={out}",
            f"--model={traj_corpus.dir / 'student.onnx'}",
            "--fast-build",
            "--rollouts=8",
            "--on-policy-min=1",
            "--on-policy-max=3",
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"gcg mode failed: {r.stderr}"
    return SimpleNamespace(set_dir=set_dir, sobs_dir=out)


def test_decision_analysis_matches_the_sidecar_and_is_plain_at_prefix_zero(gcg_set):
    """gcg_position_inputs ranks exactly the legal set the generator ranked
    (the .sobs num_legal_moves, every candidate located by bytes); the
    conditioned pass at prefix 0 equals the plain one exactly; a longer
    prefix moves the outputs; and the payload has the pane's shape."""
    from scribblez.evidence.trajectory_view import (
        DecisionAnalysis,
        payload,
        position_set_metrics,
    )
    from scribblez.ffi import gcg_position_board_json

    ckpt = _tiny_evidence_checkpoint(trained=True)
    analyses = []
    for gcg in sorted(gcg_set.set_dir.glob("*.gcg")):
        sobs = read_sobs(gcg_set.sobs_dir / f"{gcg.stem}.sobs")[0]
        a = DecisionAnalysis(ckpt, gcg.read_text(), sobs, max_e=5, device="cpu")
        analyses.append(a)
        assert a.num_legal_moves == sobs.num_legal_moves
        assert len(a.sim_index) == len(sobs.moves)
        np.testing.assert_array_equal(a.conditioned(0).value, a.plain.value)
        np.testing.assert_array_equal(a.conditioned(0).planes, a.plain.planes)
        top = max(sobs.evidence_prefix_sizes())
        if top > 0:
            assert not np.array_equal(a.conditioned(top).value, a.plain.value)

        notations = gcg_position_board_json(gcg.read_text(), open_leaves=False)["moves"]
        assert len(notations) == a.num_legal_moves
        view = payload(a, notations, top, slot=0, top_n=10)
        assert view["prefix"] == top and view["max_prefix"] == top and view["trained"]
        cards = view["trajectory"]
        assert [c["slot"] for c in cards] == list(range(len(sobs.moves)))
        assert all(c["in_prefix"] == (c["slot"] < top) for c in cards)
        assert all(c["off_policy"] == bool(sobs.roles[c["slot"]] == ROLE_OFF_POLICY) for c in cards)
        assert cards[0]["notation"] == notations[a.sim_index[0]]
        rows = {m["index"]: m for m in view["moves"]}
        assert set(a.sim_index.tolist()) <= set(rows)  # every simmed candidate is a row
        assert [m["cond_value"] for m in view["moves"]] == sorted(
            (m["cond_value"] for m in view["moves"]), reverse=True
        )
        marked = [m for m in view["moves"] if m["next_sim"]]
        if view["next_sim"] is not None:
            assert len(marked) == 1 and marked[0]["slot"] is None  # unsimmed
        planes = view["planes"]
        assert planes["slot"] == 0 and set(planes["heads"]) == {
            "opp_next_placement",
            "self_next_placement",
            "opp_win_placement",
            "self_win_placement",
        }
        assert np.asarray(planes["heads"]["opp_next_placement"]["truth"]).shape == (15, 15)
        assert payload(a, notations, top, slot=None)["planes"] is None
    metrics = position_set_metrics(analyses)
    assert metrics["posset_rows"] > 0
    assert 0.0 <= metrics["posset_cond_hit"] <= 1.0

    # Generation 0 (the student itself): conditioning is the identity at every
    # prefix and the gain column is absent.
    student = _tiny_evidence_checkpoint(trained=False)
    gcg = sorted(gcg_set.set_dir.glob("*.gcg"))[0]
    sobs = read_sobs(gcg_set.sobs_dir / f"{gcg.stem}.sobs")[0]
    a0 = DecisionAnalysis(student, gcg.read_text(), sobs, max_e=5, device="cpu")
    for p in sobs.evidence_prefix_sizes():
        np.testing.assert_array_equal(a0.conditioned(p).value, a0.plain.value)
    notations = gcg_position_board_json(gcg.read_text(), open_leaves=False)["moves"]
    view0 = payload(a0, notations, 0)
    assert not view0["trained"] and view0["next_sim"] is None
    assert all(m["gain"] is None for m in view0["moves"])
