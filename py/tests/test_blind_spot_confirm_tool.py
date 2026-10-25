"""blind_spot_confirm_tool end to end (docs/blind_spots.md, "Hidden-leaves blind
spots on the GPU"), on the CPU reference with the synthetic leaf, since the
suites hide the GPU: HastyBot games pooled by policy_set_pool_tool, then every
pool whose leader sits outside the cut confirmed into a .blindspots.json, and
the kept positions exported as .gcg files."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
POOL_TOOL = _ENGINE_DIR / "policy_set_pool_tool"
TOOL = _ENGINE_DIR / "blind_spot_confirm_tool"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file() or not TOOL.is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.policy_set.pools import read_pools  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402

CPU = ["--backend", "cpu", "--leaf-model", "synthetic", "--threads", "4"]
CUT = 3
PICKS = 2
ROLLOUTS = 40


def _confirm(slog_dir: Path, gcg_dir: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(TOOL), "--slog-dir", str(slog_dir), *CPU, "--gcg-dir", str(gcg_dir)]
    cmd += ["--cut", str(CUT), "--picks", str(PICKS), "--rollouts", str(ROLLOUTS)]
    return subprocess.run([*cmd, *flags], capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def pooled(tmp_path_factory) -> Path:
    games = tmp_path_factory.mktemp("blind") / "games"
    assert (
        run_games(games, num_games=2, threads=2, player_spec="--type=hastybot-endgame", seed=5) == 0
    )
    cmd = [str(POOL_TOOL), "--slog-dir", str(games), *CPU, "--positions-per-game", "4"]
    run = subprocess.run([*cmd, "--rollouts", "10"], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    (slog,) = games.glob("*.slog")
    return slog


def _copy(slog: Path, to: Path) -> Path:
    to.mkdir()
    for ext in (".slog", ".pool"):
        (to / slog.name).with_suffix(ext).write_bytes(slog.with_suffix(ext).read_bytes())
    return to / slog.name


@pytest.fixture(scope="module")
def confirmed(pooled, tmp_path_factory) -> tuple[Path, dict]:
    """With no bars, so every confirmed position whose outside move is no
    worse is kept."""
    gcg = tmp_path_factory.mktemp("gcg")
    run = _confirm(pooled.parent, gcg, "--min-gain", "0")
    assert run.returncode == 0, run.stderr
    return gcg, json.loads(pooled.with_name(f"{pooled.stem}.blindspots.json").read_text())


def test_header_records_the_run(pooled, confirmed):
    _, out = confirmed
    _, pools = read_pools(pooled.with_suffix(".pool"))
    assert out["pools_screened"] == len(pools)
    assert (out["cut"], out["picks"], out["rollouts"]) == (CUT, PICKS, ROLLOUTS)
    assert out["leaf_model"]["tag"] == "synthetic" and out["screen_rollouts"] == 10


def test_confirms_the_pools_led_from_outside_the_cut(pooled, confirmed):
    _, out = confirmed
    _, pools = read_pools(pooled.with_suffix(".pool"))
    led_outside = {
        (p.game, p.turn) for p in pools if not 0 <= p.equity_rank[int(np.argmax(p.q))] < CUT
    }
    assert led_outside, "the fixture has no candidate blind spot"
    assert {(p["game"], p["turn"]) for p in out["positions"]} == led_outside


def test_candidates_are_the_cut_then_the_best_picks(pooled, confirmed):
    _, out = confirmed
    pools = {(p.game, p.turn): p for p in read_pools(pooled.with_suffix(".pool"))[1]}
    for pos in out["positions"]:
        cands = pos["candidates"]
        cut = [c for c in cands if c["in_cut"]]
        picks = [c for c in cands if not c["in_cut"]]
        assert cands == cut + picks
        assert [c["equity_rank"] for c in cut] == list(range(CUT))
        assert 1 <= len(picks) <= PICKS
        pool = pools[(pos["game"], pos["turn"])]
        outside_q = np.sort(pool.q[~((pool.equity_rank >= 0) & (pool.equity_rank < CUT))])[::-1]
        np.testing.assert_allclose([c["screen_q"] for c in picks], outside_q[: len(picks)])


def test_the_verdict_compares_each_groups_best(confirmed):
    _, out = confirmed
    for pos in out["positions"]:
        wp = [c["win_prob"] for c in pos["candidates"]]
        n_cut = sum(c["in_cut"] for c in pos["candidates"])
        assert pos["best_cut"] == int(np.argmax(wp[:n_cut]))
        assert pos["best_outside"] == n_cut + int(np.argmax(wp[n_cut:]))
        np.testing.assert_allclose(pos["gain"], wp[pos["best_outside"]] - wp[pos["best_cut"]])
        assert pos["kept"] == (pos["gain"] >= 0)


def test_kept_positions_get_their_gcg(pooled, confirmed):
    gcg, out = confirmed
    kept = {
        f"{pooled.stem}-g{p['game']}-turn{p['turn'] + 1}.gcg" for p in out["positions"] if p["kept"]
    }
    assert kept and {f.name for f in gcg.iterdir()} == kept
    text = (gcg / sorted(kept)[0]).read_text()
    assert "#note hidden-leaves blind spot" in text and "beats the hasty top 3's best" in text


def test_a_high_bar_keeps_nothing(pooled, tmp_path):
    slog = _copy(pooled, tmp_path / "games")
    run = _confirm(slog.parent, tmp_path / "gcg", "--min-gain", "100")
    assert run.returncode == 0, run.stderr
    out = json.loads(slog.with_name(f"{slog.stem}.blindspots.json").read_text())
    assert out["positions"] and not any(p["kept"] for p in out["positions"])
    assert not any((tmp_path / "gcg").iterdir())


def test_a_slog_without_a_pool_is_refused(pooled, tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / pooled.name).write_bytes(pooled.read_bytes())
    run = _confirm(games, tmp_path / "gcg")
    assert run.returncode != 0 and "has no .pool" in run.stderr


def test_a_finished_file_is_skipped(pooled, confirmed):
    out = pooled.with_name(f"{pooled.stem}.blindspots.json")
    before = out.stat().st_mtime_ns
    gcg, _ = confirmed
    assert _confirm(pooled.parent, gcg).returncode == 0
    assert out.stat().st_mtime_ns == before
