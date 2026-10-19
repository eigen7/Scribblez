"""sim_candidate_survey_tool's pool recipe end to end (docs/plans/deep_bot_v0.md,
"Workload 2"): HastyBot games in, a .pool sidecar out, whose labels form a
distribution over each position's pool and whose header names the win table
the rollouts stopped at."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TOOL = _ENGINE_DIR / "sim_candidate_survey_tool"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file() or not TOOL.is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez import ffi  # noqa: E402
from scribblez.leave_inference import win_table  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402

ROLLOUTS = 20
MOVE_TYPES = ffi.format_layout()["constants"]["move_type"]
POOL = ffi.format_layout()["constants"]["pool"]


def read_pool(path: Path):
    """The .pool header and, per position, its header and candidates."""
    hdr_t = ffi.struct_dtype("PoolFileHeader")
    pos_t = ffi.struct_dtype("PoolPositionHeader")
    cand_t = ffi.struct_dtype("PoolCandidate")
    buf = np.fromfile(path, np.uint8)
    hdr = np.frombuffer(buf, hdr_t, count=1)[0]
    off, positions = hdr_t.itemsize, []
    for _ in range(int(hdr["num_positions"])):
        ph = np.frombuffer(buf, pos_t, count=1, offset=off)[0]
        off += pos_t.itemsize
        cands = np.frombuffer(buf, cand_t, count=int(ph["num_candidates"]), offset=off)
        off += cand_t.itemsize * int(ph["num_candidates"])
        positions.append((ph, cands))
    assert off == len(buf)
    return hdr, positions


@pytest.fixture(scope="module")
def pooled(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("pool")
    games = tmp / "games"
    assert (
        run_games(games, num_games=2, threads=2, player_spec="--type=hastybot-endgame", seed=5) == 0
    )
    table = tmp / "table.csv"
    rows, cols = 2 * win_table.max_spread() + 1, win_table.max_unseen() + 1
    win_table.write(table, np.full((rows, cols), 0.5), {"tag": "t", "generation": 3})
    cmd = [
        str(TOOL), "--slog-dir", str(games), "--recipe", "pool", "--positions-per-game", "1",
        "--rollouts", str(ROLLOUTS), "--win-table", str(table), "--horizon-plies", "4",
        "--threads", "4",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    (slog,) = games.glob("*.slog")
    return slog, table


def test_header_names_the_sources(pooled):
    slog, table = pooled
    hdr, positions = read_pool(slog.with_suffix(".pool"))
    assert hdr["magic"] == POOL["magic"] and hdr["version"] == POOL["version"]
    assert hdr["flags"] == 0  # uniform opponent leaves
    assert (hdr["horizon_plies"], hdr["rollouts"]) == (4, ROLLOUTS)
    assert hdr["win_table_tag"] == b"t" and hdr["win_table_generation"] == 3
    assert len(hdr["win_table_hash"]) == 16
    assert len(positions) == 2  # one per game


def test_labels_are_a_distribution_over_each_pool(pooled):
    slog, _ = pooled
    _, positions = read_pool(slog.with_suffix(".pool"))
    for ph, c in positions:
        assert len(c) == ph["num_candidates"] > 1
        np.testing.assert_allclose(c["q"].sum(), 1.0, atol=1e-5)
        assert ((c["win_prob"] >= 0) & (c["win_prob"] <= 1)).all()
        assert (c["move"]["type"] == MOVE_TYPES["pass"]).sum() == 1
        assert ((c["rollouts"] >= 1) & (c["rollouts"] <= ROLLOUTS)).all()
        leader = int(np.argmax(c["q"]))
        assert c["rollouts"][leader] == ROLLOUTS and c["paired_se"][leader] == 0
        full = c["rollouts"] == ROLLOUTS
        assert c["win_prob"][leader] == c["win_prob"][full].max()


def test_a_finished_file_is_skipped(pooled):
    slog, table = pooled
    pool = slog.with_suffix(".pool")
    before = pool.stat().st_mtime_ns
    cmd = [
        str(TOOL), "--slog-file", str(slog), "--recipe", "pool", "--rollouts", str(ROLLOUTS),
        "--win-table", str(table), "--horizon-plies", "4",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    assert pool.stat().st_mtime_ns == before
