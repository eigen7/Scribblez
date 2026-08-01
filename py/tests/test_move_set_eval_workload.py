"""Tests for the move_set_eval target-generation workload and the shared
pair-store delivery it uses."""

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.workloads import move_set_eval, pair_store
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.move_set_eval import SPEC, MoveSetEvalParams


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


def test_target_generator_command(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(move_set_eval.subprocess, "run", fake_run)
    p = MoveSetEvalParams(teacher_model="/models/teacher.onnx", quota_top=5, mid_rank_limit=24)
    slogs = [tmp_path / "a.slog", tmp_path / "b.slog"]
    assert move_set_eval.run_target_generator(slogs, p, threads=8) == 0
    cmd = captured["cmd"]
    assert cmd[0] == move_set_eval.TARGET_GENERATOR
    assert f"--slog-file={slogs[0]}" in cmd and f"--slog-file={slogs[1]}" in cmd
    assert "--model=/models/teacher.onnx" in cmd
    assert "--quota-top=5" in cmd
    assert "--mid-rank-limit=24" in cmd
    assert "--threads=8" in cmd


def test_cycle_targets_only_slogs_missing_their_sidecar(tmp_path, monkeypatch):
    # An interrupted previous cycle left one finished pair and one bare .slog;
    # the new cycle adds a fresh batch. Only the sidecar-less files get passed
    # to the generator.
    (tmp_path / "old_done.slog").touch()
    (tmp_path / "old_done.mset").touch()
    (tmp_path / "old_pending.slog").touch()

    def fake_run_games(out_dir, num_games, threads, player_spec, random_opening_mean):
        assert num_games == 200
        (out_dir / "fresh.slog").touch()
        return 0

    generated = []

    def fake_generator(pending, params, threads):
        generated.extend(p.name for p in pending)
        for p in pending:
            p.with_suffix(".mset").touch()
        return 0

    monkeypatch.setattr(move_set_eval, "run_games", fake_run_games)
    monkeypatch.setattr(move_set_eval, "run_target_generator", fake_generator)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 0
    assert sorted(generated) == ["fresh.slog", "old_pending.slog"]


def test_cycle_stops_on_selfplay_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(move_set_eval, "run_games", lambda *a, **k: 7)
    result = move_set_eval.run_one_cycle(tmp_path, MoveSetEvalParams(), threads=2)
    assert result.returncode == 7
    assert result.mset_seconds == 0.0


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
