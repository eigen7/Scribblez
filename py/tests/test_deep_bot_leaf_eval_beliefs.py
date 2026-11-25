"""deep_bot_leaf_eval's .belief sidecars (deep_bot_leaf_eval/beliefs.py): a
generation gets its .forgone and then its .belief sidecars once, a partial
set is written again, and the held-out set's store DeepBot's draws. The tools
are faked; the engine's tests cover what they write."""

import subprocess
from pathlib import Path

import pytest
from scribblez.deep_bot_leaf_eval import beliefs
from scribblez.leave_inference import forgone

MODEL = Path("/models/leave.onnx")


@pytest.fixture
def tools(monkeypatch):
    """The tool invocations, each as (tool name, its arguments); the belief
    tool's fake writes the sidecars."""
    calls = []

    def fake_run(cmd, check):
        assert check
        args = dict(zip(cmd[1::2], cmd[2::2], strict=True))
        calls.append((Path(cmd[0]).name, args))
        if Path(cmd[0]) == beliefs.TOOL:
            for slog in Path(args["--slog-dir"]).glob("*.slog"):
                slog.with_suffix(beliefs.SUFFIX).touch()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _slogs(d: Path, names=("a", "b")) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.slog").touch()
    return d


def test_a_generation_gets_its_forgone_then_its_belief_sidecars_once(tmp_path, tools):
    gen = _slogs(tmp_path / "gen_000003")
    assert beliefs.ensure_generation(gen, MODEL, threads=4)
    assert [name for name, _ in tools] == [forgone.TOOL.name, beliefs.TOOL.name]
    assert tools[1][1] == {
        "--slog-dir": str(gen),
        "--leave-inference": str(MODEL),
        "--threads": "4",
        "--draws": "0",
    }
    assert not beliefs.ensure_generation(gen, MODEL, threads=4)
    assert len(tools) == 2


def test_a_generation_missing_one_sidecar_is_written_again(tmp_path, tools):
    """A tool that died partway, or a home seeded from a copy pulled before
    the trainer wrote them."""
    gen = _slogs(tmp_path / "gen_000003")
    (gen / "a.belief").touch()
    assert beliefs.ensure_generation(gen, MODEL, threads=4)
    assert (gen / "b.belief").is_file()


def test_the_held_out_sidecars_store_deep_bots_draws(tmp_path, tools):
    held = _slogs(tmp_path / "seed1-games2000")
    assert beliefs.ensure_holdout(held, MODEL, threads=4)
    # Its .forgone sidecars are holdout.ensure's, inside its content hash.
    assert [name for name, _ in tools] == [beliefs.TOOL.name]
    assert tools[0][1]["--draws"] == str(beliefs.HOLDOUT_DRAWS)
    assert not beliefs.ensure_holdout(held, MODEL, threads=4)
