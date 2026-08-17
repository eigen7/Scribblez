"""Tests for the move_set_eval target-generation workload and the shared
pair-store delivery it uses."""

import numpy as np
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.move_set_eval import targets as T
from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP, MSET_MAGIC, MSET_VERSION
from scribblez.workloads import move_set_eval, mset_targets, pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SPEC, MoveSetEvalParams


def write_empty_pair(store, stem, flags=0):
    """A complete .slog/.mset pair whose .mset is a header and nothing else --
    all that file-level routing reads."""
    store.mkdir(parents=True, exist_ok=True)
    hdr = np.zeros(1, dtype=T._FILE_HEADER)
    hdr["magic"], hdr["version"] = MSET_MAGIC, MSET_VERSION
    hdr["record_floats"], hdr["flags"], hdr["model_hash"] = 5, flags, b"cafe"
    (store / f"{stem}.mset").write_bytes(hdr.tobytes())
    (store / f"{stem}.slog").touch()


class RecordingSink:
    kind = "local"

    def __init__(self):
        self.delivered = []  # (source file name, dest rel path), in order

    def deliver(self, src, data_rel):
        self.delivered.append((src.name, data_rel))
        src.unlink()
        return 100

    def push_json(self, rel_path, obj):
        pass


def test_workload_is_registered_with_a_valid_schema():
    spec = workloads.get("move_set_eval")
    assert spec is SPEC
    fields = {f.name: f for f in params_mod.schema(MoveSetEvalParams)}
    assert fields["teacher_model"].kind == "str"
    env = spec.worker_env("t", MoveSetEvalParams(teacher_model="/x/teacher.onnx"), "generate")
    assert env["SCZ_TEACHER_MODEL"] == "/x/teacher.onnx"
    assert params_mod.from_env(MoveSetEvalParams, env).teacher_model == "/x/teacher.onnx"


def test_train_role_is_registered():
    role = SPEC.role("train")
    assert role.singleton and role.gpu and role.kinds == ("local",)
    assert role.runner == "scribblez.move_set_eval.trainer:run"
    assert set(role.stats.phases) == {"train_s", "eval_s"}


def _stems(n, off=0, worker=0):
    """Stems as the generator emits them: a nanosecond timestamp, a worker id."""
    return [f"{1786038233456124324 + (i + off) * 4_800_000_000}-local-{worker}" for i in range(n)]


def test_split_pair_stems_is_deterministic_and_file_level():
    stems = _stems(60)
    train, holdout = move_set_eval.split_pair_stems(list(reversed(stems)), 20)
    assert holdout  # a corpus this size gets a holdout
    assert sorted(train + holdout) == sorted(stems)  # a pair is in exactly one side
    assert not set(train) & set(holdout)
    assert move_set_eval.split_pair_stems(stems, 20) == (train, holdout)  # order-free


def test_split_pair_stems_holds_out_about_one_in_n():
    stems = _stems(2000)
    _, holdout = move_set_eval.split_pair_stems(stems, 20)
    assert 0.03 < len(holdout) / len(stems) < 0.07


def test_split_pair_stems_never_moves_a_pair_between_the_sides():
    """The trainer re-takes this split as the store grows, so a pair's side has
    to be a property of the pair. Two generate workers interleave deliveries,
    so a new stem can sort BEFORE existing ones -- and a pair that changed
    sides would be one trained on and then scored as held out."""
    stems = _stems(60)
    train, holdout = move_set_eval.split_pair_stems(stems, 20)

    # A second worker's late deliveries, timestamped among the existing ones.
    grown = stems + _stems(20, off=5, worker=1)
    train2, holdout2 = move_set_eval.split_pair_stems(grown, 20)
    assert set(holdout).issubset(holdout2)
    assert set(train).issubset(train2)
    assert not set(train) & set(holdout2)  # nothing trained on became held out


def test_split_pair_stems_zero_disables_holdout():
    stems = ["b", "a", "c"]
    train, holdout = move_set_eval.split_pair_stems(stems, 0)
    assert train == ["a", "b", "c"] and holdout == []


def test_split_pairs_holds_out_the_full_sweep_pairs(tmp_path):
    """Swept pairs are the holdout wherever they exist -- they are the only
    pairs the A3 gate metrics mean anything on -- and holdout_every does not
    reserve stratified pairs on top of them."""
    for i in range(6):
        write_empty_pair(tmp_path, f"{i:03d}-local-0")
    write_empty_pair(tmp_path, "900-local-0", flags=MSET_FLAG_FULL_SWEEP)

    train, holdout = move_set_eval.split_pairs(tmp_path, holdout_every=3)
    assert [p.stem for p in holdout] == ["900-local-0"]
    assert [p.stem for p in train] == [f"{i:03d}-local-0" for i in range(6)]


def test_split_pairs_falls_back_to_holdout_every_without_sweeps(tmp_path):
    stems = _stems(60)
    for stem in stems:
        write_empty_pair(tmp_path, stem)
    (tmp_path / "orphan.mset").write_bytes(b"")  # no .slog: not a pair

    train, holdout = move_set_eval.split_pairs(tmp_path, holdout_every=20)
    expected_train, expected_holdout = move_set_eval.split_pair_stems(stems, 20)
    assert [p.stem for p in holdout] == expected_holdout
    assert [p.stem for p in train] == expected_train
    assert holdout and len(train) + len(holdout) == 60


def test_sweep_pair_is_a_stable_fraction_of_the_stems():
    stems = [f"{i:06d}-local-0" for i in range(2000)]
    swept = [s for s in stems if move_set_eval.sweep_pair(s, 20)]
    assert 0.03 < len(swept) / len(stems) < 0.07  # ~1 in 20
    # Recoverable from the file alone, which is what a resumed cycle relies on.
    assert all(move_set_eval.sweep_pair(s, 20) for s in swept)
    assert not any(move_set_eval.sweep_pair(s, 0) for s in stems)


def test_target_generator_command(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mset_targets.subprocess, "run", fake_run)
    # Distinct values per field, so a flag wired to the wrong param fails.
    p = MoveSetEvalParams(
        teacher_model="/models/teacher.onnx",
        quota_top=5,
        quota_mid=6,
        quota_tail=7,
        quota_exchange=3,
        mid_rank_limit=24,
        positions_per_game=9,
    )
    slogs = [tmp_path / "a.slog", tmp_path / "b.slog"]
    quotas = mset_targets.StratifiedQuotas.from_params(p)
    rc = mset_targets.label_stratified(slogs, p.teacher_model, quotas, p.positions_per_game, 8)
    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0] == mset_targets.TARGET_GENERATOR
    assert f"--slog-file={slogs[0]}" in cmd and f"--slog-file={slogs[1]}" in cmd
    expected = {
        "--model": "/models/teacher.onnx",
        "--quota-top": 5,
        "--quota-mid": 6,
        "--quota-tail": 7,
        "--quota-exchange": 3,
        "--mid-rank-limit": 24,
        "--positions-per-game": 9,
        "--threads": 8,
    }
    for flag, value in expected.items():
        assert f"{flag}={value}" in cmd
    assert "--full-sweep" not in cmd and "--sobs" not in cmd

    mset_targets.label_stratified(slogs, p.teacher_model, quotas, 9, 8, with_sobs=True)
    assert "--sobs" in captured["cmd"]


def test_target_generator_command_in_full_sweep_mode(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mset_targets.subprocess, "run", fake_run)
    rc = mset_targets.label_full_sweep([tmp_path / "a.slog"], "/models/teacher.onnx", 1200, 3, 8)
    assert rc == 0
    cmd = captured["cmd"]
    assert "--full-sweep" in cmd
    assert "--sweep-cap=1200" in cmd
    assert "--positions-per-game=3" in cmd
    # The two selections take disjoint parameters; the quotas are meaningless here.
    assert not any(c.startswith("--quota") for c in cmd)


class _LabelRecorder:
    """Stands in for the two labeling modes: records (mode, stems) per run and
    writes the .mset sidecars, or fails with `rc`."""

    def __init__(self, monkeypatch, rc: int = 0):
        self.runs: list[tuple[bool, set[str]]] = []
        self.rc = rc
        monkeypatch.setattr(mset_targets, "label_stratified", self._stratified)
        monkeypatch.setattr(mset_targets, "label_full_sweep", self._sweep)

    def _label(self, full_sweep, pending):
        self.runs.append((full_sweep, {p.stem for p in pending}))
        if self.rc == 0:
            for p in pending:
                p.with_suffix(".mset").touch()
        return self.rc

    def _stratified(self, pending, teacher, quotas, positions_per_game, threads, with_sobs=False):
        assert not with_sobs  # move_set_eval never force-includes candidates
        return self._label(False, pending)

    def _sweep(self, pending, teacher, cap, positions_per_game, threads):
        return self._label(True, pending)


def test_cycle_targets_only_slogs_missing_their_sidecar(tmp_path, monkeypatch):
    # An interrupted previous cycle left one finished pair and one bare .slog;
    # the new cycle adds a fresh batch. Only the sidecar-less files get passed
    # to the generator.
    (tmp_path / "old_done.slog").touch()
    (tmp_path / "old_done.mset").touch()
    (tmp_path / "old_pending.slog").touch()

    def fake_run_games(
        out_dir, num_games, threads, player_spec, random_opening_mean, face_up_leaves
    ):
        assert num_games == 200
        (out_dir / "fresh.slog").touch()
        return 0

    monkeypatch.setattr(move_set_eval, "run_games", fake_run_games)
    rec = _LabelRecorder(monkeypatch)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 0
    assert sorted(s for _, stems in rec.runs for s in stems) == ["fresh", "old_pending"]


def test_cycle_labels_each_slog_in_the_mode_its_stem_selects(tmp_path, monkeypatch):
    """The two selections run as separate generator invocations over disjoint
    files, and every pending .slog lands in exactly one of them."""
    stems = [f"{i:03d}" for i in range(60)]
    for stem in stems:
        (tmp_path / f"{stem}.slog").touch()
    expected_swept = {s for s in stems if move_set_eval.sweep_pair(s, 20)}
    assert expected_swept, "the fixture needs at least one swept stem to be meaningful"

    monkeypatch.setattr(move_set_eval, "run_games", lambda *a, **k: 0)
    rec = _LabelRecorder(monkeypatch)
    assert move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2).returncode == 0

    by_mode = dict(rec.runs)
    assert len(rec.runs) == 2
    assert by_mode[True] == expected_swept
    assert by_mode[False] == set(stems) - expected_swept


def test_cycle_stops_at_the_first_failing_selection_group(tmp_path, monkeypatch):
    """With both groups non-empty, a failure in the first must end the cycle
    with its return code -- letting the second group run would overwrite the
    failure with its own success and report a cycle that half-labeled its
    files as complete."""
    stems = [f"{i:03d}" for i in range(60)]
    for stem in stems:
        (tmp_path / f"{stem}.slog").touch()
    assert any(move_set_eval.sweep_pair(s, 20) for s in stems)
    assert not all(move_set_eval.sweep_pair(s, 20) for s in stems)

    monkeypatch.setattr(move_set_eval, "run_games", lambda *a, **k: 0)
    rec = _LabelRecorder(monkeypatch, rc=9)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 9
    assert [mode for mode, _ in rec.runs] == [False]  # the swept group never ran


def test_cycle_labels_everything_stratified_when_sweeps_are_off(tmp_path, monkeypatch):
    for i in range(40):
        (tmp_path / f"{i:03d}.slog").touch()
    monkeypatch.setattr(move_set_eval, "run_games", lambda *a, **k: 0)
    rec = _LabelRecorder(monkeypatch)
    move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(sweep_every=0), threads=2)
    assert [mode for mode, _ in rec.runs] == [False]


def test_cycle_plays_the_variant_the_params_name(tmp_path, monkeypatch):
    # The self-play condition is recorded in the .slog header and copied into
    # every .mset, so the frozen param has to reach play_game.
    seen = {}

    def fake_run_games(out_dir, **kwargs):
        seen.update(kwargs)
        (out_dir / "fresh.slog").touch()
        return 0

    monkeypatch.setattr(move_set_eval, "run_games", fake_run_games)
    _LabelRecorder(monkeypatch)
    for face_up in (False, True):
        move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(face_up_leaves=face_up), threads=2)
        assert seen["face_up_leaves"] is face_up


def test_cycle_stops_on_selfplay_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(move_set_eval, "run_games", lambda *a, **k: 7)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 7
    assert result.mset_seconds == 0.0


def test_cycle_propagates_generator_failure(tmp_path, monkeypatch):
    def fake_run_games(out_dir, **kwargs):
        (out_dir / "fresh.slog").touch()
        return 0

    monkeypatch.setattr(move_set_eval, "run_games", fake_run_games)
    _LabelRecorder(monkeypatch, rc=9)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 9


def test_run_generate_requires_a_readable_teacher(tmp_path):
    ctx = WorkerContext(
        spec=SPEC,
        role=SPEC.role("generate"),
        tag="t",
        params=MoveSetEvalParams(teacher_model=str(tmp_path / "missing.onnx")),
        worker_id="w0",
        threads=1,
        max_cycles=1,
        sink=RecordingSink(),
    )
    assert move_set_eval.run_generate(ctx) == 1


class StubCtx:
    """A WorkerContext stand-in whose tag paths live under a tmp mount."""

    def __init__(self, tmp_path, sink, max_cycles):
        self.spec = SPEC
        self.role = SPEC.role("generate")
        self.tag = "t"
        self.params = MoveSetEvalParams()
        self.worker_id = "w0"
        self.threads = 1
        self.max_cycles = max_cycles
        self.sink = sink
        self.provenance = {}
        self._paths = SPEC.paths("t", mount_root=tmp_path)

    def tag_paths(self):
        return self._paths


def test_pair_generate_loop_delivers_each_cycle(tmp_path):
    cycles = []

    def fake_cycle(work_dir, params, threads):
        cycles.append(threads)
        stem = f"c{len(cycles)}"
        (work_dir / f"{stem}.slog").write_bytes(b"s")
        (work_dir / f"{stem}.mset").write_bytes(b"m")
        return 0, {"gen_s": 0.1, "mset_s": 0.2}

    sink = RecordingSink()
    ctx = StubCtx(tmp_path, sink, max_cycles=2)
    assert pair_store.run_pair_generate(ctx, fake_cycle, ".mset", "slogs") == 0
    assert len(cycles) == 2
    assert [name for name, _rel in sink.delivered] == ["c1.mset", "c1.slog", "c2.mset", "c2.slog"]


def test_pair_generate_loop_stops_on_a_failed_cycle(tmp_path):
    def failing_cycle(work_dir, params, threads):
        return 5, {}

    ctx = StubCtx(tmp_path, RecordingSink(), max_cycles=0)
    assert pair_store.run_pair_generate(ctx, failing_cycle, ".mset", "slogs") == 5


def test_deliver_pairs_suffixes_both_members_and_leads_with_the_sidecar(tmp_path):
    for stem in ("b", "a"):
        (tmp_path / f"{stem}.slog").write_bytes(b"s")
        (tmp_path / f"{stem}.mset").write_bytes(b"m")
    (tmp_path / "orphan.slog").write_bytes(b"s")  # no sidecar: stays behind

    sink = RecordingSink()
    moved, nbytes, _ = pair_store.deliver_pairs(sink, tmp_path, "w7", ".mset", "slogs")
    assert (moved, nbytes) == (2, 400)
    assert sink.delivered == [
        ("a.mset", "slogs/a-w7.mset"),
        ("a.slog", "slogs/a-w7.slog"),
        ("b.mset", "slogs/b-w7.mset"),
        ("b.slog", "slogs/b-w7.slog"),
    ]
    assert [p.name for p in tmp_path.iterdir()] == ["orphan.slog"]


def test_count_pairs(tmp_path):
    assert pair_store.count_pairs(tmp_path / "absent", ".mset") == 0
    (tmp_path / "a.mset").touch()
    (tmp_path / "a.slog").touch()
    (tmp_path / "b.slog").touch()
    assert pair_store.count_pairs(tmp_path, ".mset") == 1


class StoringSink:
    """A sink that lands pairs in the tag's own data tree, as the local sink
    does -- which is what a generation target reads to know when to stop."""

    kind = "local"

    def __init__(self, root):
        self.root = root

    def deliver(self, src, data_rel):
        dest = self.root / data_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        return dest.stat().st_size

    def push_json(self, rel_path, obj):
        pass


def test_pair_generate_stops_once_the_store_holds_the_target(tmp_path):
    """A tag with a generation size finishes on its own: the target is read off
    the store, so it counts what the tag holds rather than what this worker
    made."""
    cycles = []

    def fake_cycle(work_dir, params, threads):
        cycles.append(1)
        stem = f"c{len(cycles)}"
        (work_dir / f"{stem}.slog").write_bytes(b"s")
        (work_dir / f"{stem}.mset").write_bytes(b"m")
        return 0, {"gen_s": 0.1, "mset_s": 0.2}

    ctx = StubCtx(tmp_path, None, max_cycles=0)  # unbounded but for the target
    ctx.sink = StoringSink(ctx.tag_paths().data_dir)
    assert pair_store.run_pair_generate(ctx, fake_cycle, ".mset", "slogs", target_pairs=3) == 0
    assert len(cycles) == 3
    assert pair_store.count_pairs(ctx.tag_paths().data_dir / "slogs", ".mset") == 3

    # Restarting against a store already at the target does no work at all.
    assert pair_store.run_pair_generate(ctx, fake_cycle, ".mset", "slogs", target_pairs=3) == 0
    assert len(cycles) == 3


def test_pair_generate_without_a_target_is_unbounded(tmp_path):
    """The shared loop is used by workloads that declare no size; they must
    keep the run-until-paused behavior, and must not have their store counted."""
    calls = []

    def fake_cycle(work_dir, params, threads):
        calls.append(1)
        stem = f"c{len(calls)}"
        (work_dir / f"{stem}.slog").write_bytes(b"s")
        (work_dir / f"{stem}.mset").write_bytes(b"m")
        return 0, {}

    ctx = StubCtx(tmp_path, RecordingSink(), max_cycles=4)
    assert pair_store.run_pair_generate(ctx, fake_cycle, ".mset", "slogs") == 0
    assert len(calls) == 4
