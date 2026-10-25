"""The hidden_blind_spots workload: a cycle on the tools' CPU reference with
the synthetic leaf (the suites hide the GPU), what it delivers, and the
progress counts. The scheduler hooks are blind_spots', tested there."""

from dataclasses import replace
from pathlib import Path

import pytest
from scribblez.paths import ENGINE_DIR
from scribblez.workloads import hidden_blind_spots as hbs
from scribblez.workloads.policy_set import BELIEFS_UNIFORM, LEAF_SYNTHETIC
from tests.test_blind_spots_workload import RecordingSink, StubPaths

if not hbs.CONFIRM_TOOL.is_file() or not (ENGINE_DIR / "policy_set_pool_tool").is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)

PARAMS = hbs.HiddenBlindSpotsParams(
    leave_beliefs=BELIEFS_UNIFORM,
    rollouts=10,
    positions_per_game=4,
    games_per_chunk=2,
    cut=3,
    confirm_rollouts=40,
    min_gain=0.0,
)


@pytest.fixture(scope="module")
def confirmed_chunk(tmp_path_factory) -> Path:
    chunk = tmp_path_factory.mktemp("cycle")
    hbs.play_chunk(chunk, "w1", PARAMS, threads=2)
    hbs.confirm_chunk(PARAMS, LEAF_SYNTHETIC, None, chunk, threads=4)
    return chunk


def test_a_chunk_is_named_for_its_worker(confirmed_chunk):
    (slog,) = confirmed_chunk.glob("*.slog")
    assert slog.stem.endswith("-w1")
    assert slog.with_suffix(".pool").exists()
    assert slog.with_name(f"{slog.stem}{hbs.FOUND_SUFFIX}").exists()


def test_a_cycle_delivers_the_games_the_report_and_the_kept_positions(confirmed_chunk, tmp_path):
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    for f in confirmed_chunk.rglob("*"):
        if f.is_file():
            dest = chunk / f.relative_to(confirmed_chunk)
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(f.read_bytes())
    (slog,) = chunk.glob("*.slog")
    gcgs = sorted(f.name for f in (chunk / hbs.GCG_DIR).glob("*.gcg"))
    assert gcgs, "no zero-bar position was kept"
    sink = RecordingSink()
    kept, nbytes = hbs.deliver_chunk(sink, chunk)
    assert kept == len(gcgs) and nbytes > 0
    # The games and the report first, so the counted .gcg never arrives
    # without them; no .forgone under uniform beliefs, and never the pool.
    assert sink.delivered == [
        f"found/{slog.name}",
        f"found/{slog.stem}{hbs.FOUND_SUFFIX}",
        *[f"gcg/{g}" for g in gcgs],
    ]
    assert slog.with_suffix(".pool").exists()  # left for the cycle's cleanup


def test_the_confirm_command_carries_the_params(tmp_path):
    p = replace(PARAMS, cut=7, picks=3, confirm_rollouts=900, min_gain=4.5)
    cmd = hbs.confirm_command(p, tmp_path, "leaf.onnx", Path("li.onnx"), seed=9, threads=3)
    flags = dict(zip(cmd[1::2], cmd[2::2], strict=False))
    assert flags["--cut"] == "7" and flags["--picks"] == "3" and flags["--rollouts"] == "900"
    assert flags["--min-gain"] == "4.5"
    assert flags["--leave-inference"] == "li.onnx" and "--backend" not in flags
    assert flags["--gcg-dir"] == str(tmp_path / hbs.GCG_DIR)


def test_progress_counts_the_tags_store(tmp_path):
    (tmp_path / hbs.GCG_DIR).mkdir()
    (tmp_path / hbs.FOUND_DIR).mkdir()
    for name in ("a.gcg", "b.gcg"):
        (tmp_path / hbs.GCG_DIR / name).touch()
    (tmp_path / hbs.FOUND_DIR / f"a{hbs.FOUND_SUFFIX}").touch()
    (tmp_path / hbs.FOUND_DIR / "a.slog").touch()
    counts = hbs.progress(hbs.SPEC, StubPaths(tmp_path), None)
    assert counts == [("positions kept", 2), ("chunks confirmed", 1)]
