"""Tests for the sim-evidence data path: sim_obs_tool -> .sobs parsing ->
row decoding by identity -> evidence features -> the evidence-conditioned
model. The end-to-end pieces run the real C++ tool on freshly generated
.slog fixtures and are skipped when the binary or the HastyBot leave values
(mount macondo strategy data) are unavailable.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.dataset import row_layout
from scribblez.evidence.dataset import TrajectoryDataset
from scribblez.ffi import decode_rows, get_input_shapes, row_size_floats
from scribblez.footprint_spatial import PASS_CLASS, SLOTS_PER_CELL
from scribblez.sim_evidence.model import NUM_EVIDENCE_PLANES, EvidencePositionEvalModel
from scribblez.sim_evidence.sobs import (
    _FILE_HEADER,
    MOVE_EXCHANGE,
    MOVE_PASS,
    MOVE_PLAY,
    NUM_EVIDENCE_SCALARS,
    RECORD_DTYPE,
    SOBS_FLAG_TRAJECTORY,
    SOBS_MAGIC,
    SOBS_VERSION,
    evidence_features,
    move_footprint,
    move_footprint_class,
    read_sobs,
)

# This checkout's own binaries and data, not the primary checkout's -- see the
# note in test_move_set_eval_targets.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Frozen .gcg fixtures. Live position sets under positions/ are regenerated
# and renamed; a test must never read one.
TEST_DATA = _REPO_ROOT / "engine" / "tests" / "data"
SIM_OBS_TOOL = _REPO_ROOT / "target" / "engine" / "sim_obs_tool"
SLOG_WRITER = _REPO_ROOT / "target" / "engine" / "test_slog_writer"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

ROLLOUTS = 8
TOP_K = 3


@pytest.fixture(scope="module")
def sobs_dir(tmp_path_factory) -> Path:
    """A directory of .slog files with freshly generated .sobs sidecars."""
    if not SIM_OBS_TOOL.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    d = tmp_path_factory.mktemp("sim_evidence")
    # Four games is what the assertions below read: several files to iterate,
    # each with enough turns that the sampled positions span opening and
    # midgame. The tool's cost is linear in the corpus, so a larger one buys
    # runtime rather than coverage.
    subprocess.run([str(SLOG_WRITER), str(d), "4", "6"], check=True, capture_output=True)
    result = subprocess.run(
        [
            str(SIM_OBS_TOOL),
            f"--slog-dir={d}",
            f"--rollouts={ROLLOUTS}",
            f"--top-k={TOP_K}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"sim_obs_tool failed: {result.stderr}"
    assert sorted(d.glob("*.sobs")), "no .sobs produced"
    return d


def test_slog_file_mode_matches_dir_mode(sobs_dir):
    # Regenerating one sidecar via an explicit --slog-file must reproduce the
    # directory-mode output byte for byte (same run seed drives both).
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    sobs = slog.with_suffix(".sobs")
    original = sobs.read_bytes()
    sobs.unlink()
    result = subprocess.run(
        [
            str(SIM_OBS_TOOL),
            f"--slog-file={slog}",
            f"--rollouts={ROLLOUTS}",
            f"--top-k={TOP_K}",
            "--positions-per-game=2",
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"sim_obs_tool failed: {result.stderr}"
    assert sobs.read_bytes() == original


def test_sobs_positions_parse_and_hold_invariants(sobs_dir):
    total_positions = 0
    for sobs in sorted(sobs_dir.glob("*.sobs")):
        positions = read_sobs(sobs)
        total_positions += len(positions)
        for pos in positions:
            assert pos.rollouts == ROLLOUTS
            assert 1 <= len(pos.moves) <= TOP_K
            for move, obs in zip(pos.moves, pos.obs, strict=True):
                assert int(move["type"]) in (0, 1, 2)
                assert int(obs["n"]) == ROLLOUTS
                assert int(obs["wins"]) + int(obs["draws"]) + int(obs["losses"]) == ROLLOUTS
                assert (obs["opp_win_count"] <= obs["opp_next_count"]).all()
                assert (obs["self_win_count"] <= obs["self_next_count"]).all()
                assert (obs["opp_next_count"] <= ROLLOUTS).all()
    assert total_positions > 0


def test_sobs_positions_decode_as_training_rows(sobs_dir):
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    positions = read_sobs(slog.with_suffix(".sobs"))
    games = np.array([p.game_index for p in positions], dtype=np.int64)
    turns = np.array([p.turn_index for p in positions], dtype=np.int64)
    rows = decode_rows(slog, games, turns, post_move=True)
    assert rows.shape == (len(positions), row_size_floats())
    # The WLD one-hot (located via the row layout): exactly one of 3 set per row.
    _, targets = row_layout()
    start, end = next((s, e) for name, s, e, _ in targets if name == "wld")
    wld = rows[:, start:end]
    assert np.all(wld.sum(axis=1) == 1.0)


def _play_record(horizontal: bool, start: int, along0: int, k: int) -> np.void:
    """One MOVE_DTYPE play of `k` contiguous tiles: lane `start`, first placed
    lane cell `along0`."""
    move = np.zeros(1, dtype=RECORD_DTYPE["move"])[0]
    move["type"] = MOVE_PLAY
    move["horizontal"] = int(horizontal)
    move["start"] = start
    move["square_mask"] = sum(1 << (along0 + t) for t in range(k))
    move["num_played"] = k
    move["glyphs"][:k] = 1
    return move


def test_move_footprint_class_matches_the_footprint_layout():
    """Hand-computed classes across orientations and tile counts, pinning the
    numpy mirror of the C++ footprint_class: slot 0 is the orientation-free
    k==1 footprint, 1..6 horizontal k=2..7, 7..12 vertical k=2..7, and
    class = (r*15 + c)*13 + slot at the anchor (first placed square). The
    C++/Python parity test crosses the two implementations end to end; this
    pins the layout directly, case by case."""
    max_k = (SLOTS_PER_CELL + 1) // 2  # 7
    cases = [
        # (horizontal, start, along0, k) -> (anchor r, anchor c, slot)
        (True, 3, 6, 1, 3, 6, 0),
        (False, 6, 3, 1, 3, 6, 0),  # k==1 is orientation-free: same class
        (True, 3, 6, 2, 3, 6, 1),
        (True, 0, 0, 7, 0, 0, 6),  # horizontal slots run 1..6
        (False, 5, 2, 2, 2, 5, max_k),  # vertical base
        (False, 5, 2, 3, 2, 5, max_k + 1),
        (False, 14, 8, 7, 8, 14, 2 * max_k - 2),  # last slot, 12
    ]
    for horizontal, start, along0, k, r, c, slot in cases:
        move = _play_record(horizontal, start, along0, k)
        expected = (r * 15 + c) * SLOTS_PER_CELL + slot
        assert move_footprint_class(move) == expected, (horizontal, start, along0, k)
    # Non-plays map to the pass catch-all.
    for mtype in (MOVE_EXCHANGE, MOVE_PASS):
        move = np.zeros(1, dtype=RECORD_DTYPE["move"])[0]
        move["type"] = mtype
        assert move_footprint_class(move) == PASS_CLASS


def test_move_footprint_and_features(sobs_dir):
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    positions = read_sobs(slog.with_suffix(".sobs"))
    pos = positions[0]
    obs_channels = 4 * SLOTS_PER_CELL
    planes, scalars, mask = evidence_features(pos, max_k=TOP_K + 2)
    assert planes.shape == (TOP_K + 2, 5 * SLOTS_PER_CELL, 15, 15)
    assert scalars.shape == (TOP_K + 2, NUM_EVIDENCE_SCALARS)
    assert mask.sum() == len(pos.moves)
    for i, move in enumerate(pos.moves):
        # The per-cell mirror of visit_placed_squares still walks the squares.
        foot = move_footprint(move)
        if int(move["type"]) == MOVE_PLAY:
            assert foot.sum() == int(move["num_played"])
        else:
            assert foot.sum() == 0
        # The candidate block is a one-hot at (footprint slot, anchor cell)
        # for a play, all zeros for a pass/exchange (dropped catch-all).
        cand = planes[i, obs_channels:].astype(np.float32)
        if int(move["type"]) == MOVE_PLAY:
            cls = move_footprint_class(move)
            cell, slot = divmod(cls, SLOTS_PER_CELL)
            assert cand.sum() == 1.0
            assert cand[slot, cell // 15, cell % 15] == 1.0
            assert foot[cell // 15, cell % 15] == 1.0  # the anchor is a placed square
        else:
            assert cand.sum() == 0
        # Frequencies are per-rollout fractions in [0, 1]; each observed head's
        # anchored mass is at most its rollout total.
        assert float(planes[i, :obs_channels].max()) <= 1.0
        assert float(planes[i, :SLOTS_PER_CELL].astype(np.float32).sum()) <= 1.0 + 1e-3
    # W/D/L frequencies sum to 1 for real candidates.
    assert np.allclose(scalars[mask][:, :3].sum(axis=1), 1.0)


def test_gcg_sim_evidence_on_dataset_position():
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    from scribblez.ffi import gcg_sim_evidence

    gcg = TEST_DATA / "postbingo-gave.gcg"
    records, played_rank = gcg_sim_evidence(gcg.read_text(), top_k=4, rollouts=8, threads=4, seed=3)
    assert 1 <= len(records) <= 4
    assert -1 <= played_rank < len(records)
    for rec in records:
        obs = rec["obs"]
        assert int(obs["n"]) == 8
        assert int(obs["wins"]) + int(obs["draws"]) + int(obs["losses"]) == 8
        assert (obs["opp_win_count"] <= obs["opp_next_count"]).all()
    # Determinism: the same seed reproduces the records byte for byte.
    records2, rank2 = gcg_sim_evidence(gcg.read_text(), top_k=4, rollouts=8, threads=2, seed=3)
    assert rank2 == played_rank
    assert records2.tobytes() == records.tobytes()


def test_open_leaves_sobs_and_input_arm(sobs_dir, tmp_path):
    import sys

    from scribblez.sim_evidence.sobs import (
        SOBS_FLAG_OPEN_LEAVES,
        read_sobs_flags,
        read_sobs_leaf,
    )

    # Hidden-mode sidecars carry no flags, and terminal sims no leaf model.
    assert read_sobs_flags(sorted(sobs_dir.glob("*.sobs"))[0]) == 0
    assert read_sobs_leaf(sorted(sobs_dir.glob("*.sobs"))[0]) == ("", 0)

    # Regenerate one sidecar open-leaves: the flag is recorded and the sims
    # still satisfy the observation invariants.
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    import shutil

    slog_copy = tmp_path / slog.name
    shutil.copy(slog, slog_copy)
    result = subprocess.run(
        [
            str(SIM_OBS_TOOL),
            f"--slog-file={slog_copy}",
            "--open-leaves",
            f"--rollouts={ROLLOUTS}",
            f"--top-k={TOP_K}",
            "--positions-per-game=2",
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    sobs = slog_copy.with_suffix(".sobs")
    assert read_sobs_flags(sobs) == SOBS_FLAG_OPEN_LEAVES
    for pos in read_sobs(sobs):
        for obs in pos.obs:
            assert int(obs["n"]) == ROLLOUTS
            assert (obs["opp_win_count"] <= obs["opp_next_count"]).all()

    # The open-leaves input arm appends the 27 opponent-leave counts. The FFI
    # session's layout is fixed at creation, so probe it in a subprocess.
    code = (
        "from scribblez.ffi import set_opp_leave_input, get_input_shapes\n"
        "set_opp_leave_input(True)\n"
        "print({s.name: s.dims for s in get_input_shapes()}['input_scalar'][0])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    base = {s.name: s.dims for s in get_input_shapes()}["input_scalar"][0]
    assert int(out.stdout.strip()) == base + 27


def test_position_meta_flags(sobs_dir):
    from scribblez.sim_evidence.slog_meta import game_metas, move_at, position_meta, read_slog_bytes

    slog = sorted(sobs_dir.glob("*.slog"))[0]
    positions = read_sobs(slog.with_suffix(".sobs"))
    meta = position_meta(slog, positions)
    assert len(meta["meta_turn"]) == len(positions)
    buf = read_slog_bytes(slog)
    metas = game_metas(buf)
    for i, pos in enumerate(positions):
        assert meta["meta_turn"][i] == pos.turn_index
        num_turns = int(metas[pos.game_index]["num_turns"])
        assert meta["meta_remaining"][i] == num_turns - pos.turn_index
        if pos.turn_index == 0:
            assert meta["meta_opp_unbiased"][i]
        else:
            prev = move_at(buf, metas[pos.game_index], pos.turn_index - 1)
            expect = int(prev["type"]) == MOVE_PLAY and int(prev["num_played"]) == 7
            assert meta["meta_opp_unbiased"][i] == expect


def _tiny_model() -> EvidencePositionEvalModel:
    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    return EvidencePositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=16,
        num_blocks=2,
    )


def test_model_ignores_masked_evidence():
    model = _tiny_model().eval()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    b, k = 2, 4
    spatial = torch.randn(b, *shapes["input_spatial"])
    scalar = torch.randn(b, shapes["input_scalar"][0])
    mask = torch.zeros(b, k, dtype=torch.bool)
    with torch.no_grad():
        out_zero = model(
            spatial,
            scalar,
            torch.zeros(b, k, NUM_EVIDENCE_PLANES, 15, 15),
            torch.zeros(b, k, NUM_EVIDENCE_SCALARS),
            mask,
        )
        out_junk = model(
            spatial,
            scalar,
            torch.randn(b, k, NUM_EVIDENCE_PLANES, 15, 15),
            torch.randn(b, k, NUM_EVIDENCE_SCALARS),
            mask,
        )
    for key in out_zero:
        assert torch.equal(out_zero[key], out_junk[key]), key


def test_model_trains_on_evidence():
    model = _tiny_model()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    b, k = 3, 4
    spatial = torch.randn(b, *shapes["input_spatial"])
    scalar = torch.randn(b, shapes["input_scalar"][0])
    planes = torch.rand(b, k, NUM_EVIDENCE_PLANES, 15, 15)
    scalars = torch.randn(b, k, NUM_EVIDENCE_SCALARS)
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[0, 2:] = False  # a padded set in the batch
    out = model(spatial, scalar, planes, scalars, mask)
    out["wld"].sum().backward()
    # The fusion projections are zero-initialized, so at init they (not the
    # encoder behind them) are the learning conduits: their weight gradients
    # must be nonzero, which is what pulls them off zero and opens gradient
    # flow into the rest of the evidence encoder.
    for param in (model.evidence.spatial_out.weight, model.value_proj.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0


def test_empty_evidence_is_finite():
    # An all-padding evidence set exercises the empty-attention guard.
    model = _tiny_model().eval()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial = torch.randn(1, *shapes["input_spatial"])
    scalar = torch.randn(1, shapes["input_scalar"][0])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with torch.no_grad():
        out = model(
            spatial,
            scalar,
            torch.zeros(1, 3, NUM_EVIDENCE_PLANES, 15, 15),
            torch.zeros(1, 3, NUM_EVIDENCE_SCALARS),
            mask,
        )
    for value in out.values():
        assert torch.isfinite(value).all()


def _write_header_only_sobs(path: Path, *, proposer: str, leaf_hash: str, horizon: int) -> Path:
    """A valid trajectory .sobs holding no positions -- enough to exercise
    TrajectoryDataset.absorb's header-consistency checks without an engine
    binary."""
    hdr = np.zeros(1, dtype=_FILE_HEADER)
    hdr["magic"] = SOBS_MAGIC
    hdr["version"] = SOBS_VERSION
    hdr["flags"] = SOBS_FLAG_TRAJECTORY
    hdr["horizon_plies"] = horizon
    hdr["num_positions"] = 0
    hdr["proposer_hash"] = proposer.encode()
    hdr["leaf_model_hash"] = leaf_hash.encode()
    path.write_bytes(hdr.tobytes())
    path.with_suffix(".slog").touch()  # absorb records the sibling; never read here
    return path


def test_absorb_refuses_mismatched_leaf_hash(tmp_path):
    a = _write_header_only_sobs(tmp_path / "a.sobs", proposer="p", leaf_hash="cafe", horizon=4)
    b = _write_header_only_sobs(tmp_path / "b.sobs", proposer="p", leaf_hash="beef", horizon=4)
    with pytest.raises(ValueError, match="leaf models/horizons"):
        TrajectoryDataset([a, b])


def test_absorb_refuses_mismatched_horizon(tmp_path):
    a = _write_header_only_sobs(tmp_path / "a.sobs", proposer="p", leaf_hash="cafe", horizon=4)
    b = _write_header_only_sobs(tmp_path / "b.sobs", proposer="p", leaf_hash="cafe", horizon=6)
    with pytest.raises(ValueError, match="leaf models/horizons"):
        TrajectoryDataset([a, b])


def test_absorb_accepts_matching_leaf(tmp_path):
    a = _write_header_only_sobs(tmp_path / "a.sobs", proposer="p", leaf_hash="cafe", horizon=4)
    b = _write_header_only_sobs(tmp_path / "b.sobs", proposer="p", leaf_hash="cafe", horizon=4)
    ds = TrajectoryDataset([a, b])
    assert ds.num_positions == 0
    assert ds.files == [a, b]
