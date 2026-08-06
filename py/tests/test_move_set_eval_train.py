"""Tests for the move set evaluation model training side: the move-feature encoder
(pure CPU) and the end-to-end dataset -> train step -> recall eval path over a
tiny generated .mset/.slog corpus (GPU, since the target generator builds a
TensorRT engine).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.move_set_eval import moves as move_enc
from scribblez.sim_evidence.sobs import MOVE_DTYPE, MOVE_PLAY, move_footprint

# This checkout's own binaries, so a worktree's tests exercise the code built
# beside them rather than the primary checkout's (as in test_move_set_eval_targets.py).
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TARGET_GENERATOR = _ENGINE_DIR / "move_set_eval_target_generator"
SLOG_WRITER = _ENGINE_DIR / "test_slog_writer"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

QUOTAS = {"top": 3, "mid": 2, "tail": 2, "exchange": 1}


def test_encode_moves_matches_footprint():
    """The move encoder's placed squares agree with the .sobs footprint walk,
    and the scalar block reflects the resultant differential / tile-count /
    is-play."""
    move = np.zeros(1, dtype=MOVE_DTYPE)[0]
    move["type"] = MOVE_PLAY
    move["horizontal"] = 1
    move["start"] = 7
    move["square_mask"] = (1 << 7) | (1 << 9) | (1 << 10)
    move["num_played"] = 3
    move["glyphs"][:3] = [18, 1, 4]  # R, A, D (natural letters)
    move["score"] = 20

    # Pre-move differential of 30; resultant = 30 + 20 = 50 -> 50/100 = 0.5.
    enc = move_enc.encode_moves(np.array([move], dtype=MOVE_DTYPE), np.array([30], dtype=np.int32))
    mask = enc["tile_mask"][0]
    assert mask.sum() == 3
    squares = enc["squares"][0][mask]
    got = {(int(s) // move_enc.BOARD, int(s) % move_enc.BOARD) for s in squares}
    footprint = move_footprint(move)
    expected = set(zip(*np.nonzero(footprint), strict=True))
    assert got == expected
    # Natural letters: id == A..Z index + 1, so R/A/D -> 18/1/4; no blanks.
    assert list(enc["letters"][0][mask]) == [18, 1, 4]
    assert not enc["blanks"][0][mask].any()
    np.testing.assert_allclose(enc["scalars"][0], [0.50, 3 / 7, 1.0], atol=1e-6)


def _ragged_batch(counts: list[int], seed: int = 0) -> dict[str, torch.Tensor]:
    """A random flattened candidate batch shaped the way the dataset emits one:
    per-position blocks concatenated in position order, `counts[i]` moves for
    position i. Float inputs are float64 so an equivalence check can be tight."""
    from scribblez.dataset import row_layout

    input_shapes, _ = row_layout()
    spatial_shape = tuple(input_shapes[0].dims)
    scalar_size = int(input_shapes[1].dims[0])
    max_tiles, num_scalars, letter_vocab, cells = move_enc.move_encoding_dims()

    p, m = len(counts), sum(counts)
    gen = torch.Generator().manual_seed(seed)
    return {
        "input_spatial": torch.rand(p, *spatial_shape, generator=gen, dtype=torch.float64),
        "input_scalar": torch.rand(p, scalar_size, generator=gen, dtype=torch.float64),
        "move_letters": torch.randint(0, letter_vocab, (m, max_tiles), generator=gen),
        "move_blanks": torch.randint(0, 2, (m, max_tiles), generator=gen),
        "move_squares": torch.randint(0, cells, (m, max_tiles), generator=gen),
        "move_tile_mask": (torch.rand(m, max_tiles, generator=gen) > 0.4).double(),
        "move_scalars": torch.randn(m, num_scalars, generator=gen, dtype=torch.float64),
        "move_pos_id": torch.repeat_interleave(torch.arange(p), torch.tensor(counts)),
    }


_BOARD_KEYS = ("input_spatial", "input_scalar")
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
)
# MoveSetEvalModel.forward's positional order.
_FORWARD_KEYS = (*_BOARD_KEYS, *_MOVE_KEYS, "move_pos_id")


def test_scoring_is_independent_across_positions():
    """A candidate's outputs depend only on its own features and its own
    position's board -- never on which other candidates share the batch.

    The cross-attention packs the flattened move set into a padded (P, maxK)
    query grid so the board's key/value projections run once per position, and
    this independence is what makes that safe: queries never attend to each
    other, so the pad rows cannot reach the real ones and the ragged counts
    cannot leak across positions. Scoring a ragged batch in one call must
    therefore reproduce scoring each of its positions alone.
    """
    from scribblez.move_set_eval.model import MoveSetEvalModel

    counts = [1, 5, 12, 3, 8]  # ragged, including the maxK position and a singleton
    batch = _ragged_batch(counts)
    torch.manual_seed(0)
    model = MoveSetEvalModel(
        spatial_planes=batch["input_spatial"].shape[1],
        scalar_size=batch["input_scalar"].shape[1],
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    model = model.double().eval()

    with torch.no_grad():
        joint = model(*(batch[k] for k in _FORWARD_KEYS))

    start = 0
    for i, k in enumerate(counts):
        moves = slice(start, start + k)
        solo_args = [batch[key][i : i + 1] for key in _BOARD_KEYS]
        solo_args += [batch[key][moves] for key in _MOVE_KEYS]
        solo_args.append(torch.zeros(k, dtype=torch.int64))  # all moves on row 0
        with torch.no_grad():
            solo = model(*solo_args)
        for name, value in solo.items():
            torch.testing.assert_close(value, joint[name][moves], rtol=0, atol=1e-12)
        start += k


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    """A tiny .slog corpus labeled with .mset targets from a small teacher."""
    if not TARGET_GENERATOR.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from scribblez.ffi import get_input_shapes
    from scribblez.position_eval.model import PositionEvalModel
    from scribblez.position_eval.onnx_export import export_onnx

    d = tmp_path_factory.mktemp("move_set_eval_train")
    subprocess.run([str(SLOG_WRITER), str(d), "12", "4"], check=True, capture_output=True)

    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    teacher = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=3,
    ).eval()
    onnx_path = d / "teacher.onnx"
    export_onnx(
        teacher,
        onnx_path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
        opp_leave_input=False,
    )
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={d}",
            f"--model={onnx_path}",
            "--fast-build",
            f"--quota-top={QUOTAS['top']}",
            f"--quota-mid={QUOTAS['mid']}",
            f"--quota-tail={QUOTAS['tail']}",
            f"--quota-exchange={QUOTAS['exchange']}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"target generator failed: {result.stderr}"
    assert sorted(d.glob("*.mset")), "no .mset produced"
    return d


@pytest.fixture(scope="module")
def sweep_dir(corpus_dir, tmp_path_factory) -> Path:
    """The same games labeled in the generator's full-sweep mode: the held-out
    evaluation slice, whose positions carry hundreds of candidates each."""
    d = tmp_path_factory.mktemp("move_set_eval_sweep")
    for slog in sorted(corpus_dir.glob("*.slog")):
        shutil.copy(slog, d / slog.name)
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={d}",
            f"--model={corpus_dir / 'teacher.onnx'}",
            "--fast-build",
            "--full-sweep",
            "--sweep-cap=400",
            "--positions-per-game=1",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"full-sweep generator failed: {result.stderr}"
    return d


def test_eval_runs_over_a_full_sweep_holdout(sweep_dir):
    """The A3 gate path end to end: a swept holdout is far larger per position
    than a stratified one, so this exercises the candidate-budget batching and
    the metrics over full legal sets."""
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.eval import evaluate
    from scribblez.move_set_eval.model import MoveSetEvalModel

    ds = MsetDataset(sweep_dir)
    assert ds.full_sweep
    assert ds.num_candidates / ds.num_positions > 20  # nothing like a 15-candidate sample
    coverage, _ = ds.sweep_coverage
    assert 0.0 < coverage <= 1.0

    torch.manual_seed(0)
    model = MoveSetEvalModel(
        spatial_planes=ds.spatial_planes,
        scalar_size=ds.scalar_size,
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    metrics = evaluate(
        model, ds, torch.device("cpu"), positions_per_batch=64, max_candidates_per_batch=256
    )
    assert metrics["positions"] == ds.num_positions
    for k in (1, 3, 5):
        assert 0.0 <= metrics[f"recall@{k}"] <= 1.0
        assert metrics[f"regret@{k}"] >= 0.0
    # The baseline over a sweep is the exact static-equity ranking, which is a
    # real move ordering rather than a shuffle: it must beat a coin flip.
    assert metrics["spearman_baseline"] > 0.0


def test_dataset_batches_flatten_candidates(corpus_dir):
    from scribblez.move_set_eval.dataset import MsetDataset

    ds = MsetDataset(corpus_dir)
    assert ds.num_positions > 0
    assert ds.num_candidates >= ds.num_positions  # >= 1 candidate each

    seen_positions = 0
    for batch in ds.iter_batches(positions_per_batch=8, seed=0):
        p = batch["input_spatial"].shape[0]
        m = batch["move_pos_id"].shape[0]
        assert batch["input_spatial"].shape[1] == ds.spatial_planes
        assert batch["input_scalar"].shape == (p, ds.scalar_size)
        # Flattened move tensors are all length M and map into [0, P).
        move_keys = ("move_letters", "move_blanks", "move_squares", "move_tile_mask")
        for key in (*move_keys, "move_scalars"):
            assert batch[key].shape[0] == m
        assert int(batch["move_pos_id"].max()) < p
        assert batch["target_wld"].shape == (m, 3)
        assert batch["target_score_diff"].shape == (m, 2)
        seen_positions += p
    assert seen_positions == ds.num_positions


def test_train_step_and_eval(corpus_dir):
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.eval import evaluate
    from scribblez.move_set_eval.model import MoveSetEvalModel
    from scribblez.move_set_eval.train_loop import LossConfig, run_epoch

    device = torch.device("cpu")
    ds = MsetDataset(corpus_dir)
    torch.manual_seed(0)
    model = MoveSetEvalModel(
        spatial_planes=ds.spatial_planes,
        scalar_size=ds.scalar_size,
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_cfg = LossConfig(lambda_sd=0.004, huber_delta_mean=10.0, huber_delta_std=10.0)

    rows = 0
    first = last = None
    for epoch in range(3):
        result = run_epoch(
            model,
            optimizer,
            ds.iter_batches(positions_per_batch=8, seed=0, epoch_index=epoch),
            device,
            loss_cfg,
            rows_trained=rows,
        )
        rows = result.rows_trained
        assert np.isfinite(result.losses["total"])
        first = result.losses["total"] if first is None else first
        last = result.losses["total"]
    assert rows == 3 * ds.num_candidates
    assert last <= first * 2.0  # a few steps should not diverge

    metrics = evaluate(model, ds, device, positions_per_batch=8)
    assert metrics["positions"] == ds.num_positions
    for k in (1, 3, 5):
        for suffix in ("", "_baseline"):
            assert 0.0 <= metrics[f"recall@{k}{suffix}"] <= 1.0
            assert metrics[f"regret@{k}{suffix}"] >= 0.0
    # Larger K keeps a superset of the candidates, so regret can only shrink.
    assert metrics["regret@5"] <= metrics["regret@3"] <= metrics["regret@1"]
    for suffix in ("", "_baseline"):
        assert -1.0 <= metrics[f"spearman{suffix}"] <= 1.0


def _write_mset(path, positions, flags=0):
    """Hand-pack a minimal v1 .mset (layout mirrored by targets.py's dtypes),
    plus its companion .slog. A position is (game_index, turn_index, targets),
    optionally followed by the swept position's legal-move count."""
    from scribblez.move_set_eval import targets as T

    parts = []
    hdr = np.zeros(1, dtype=T._FILE_HEADER)
    hdr["magic"], hdr["version"] = T.MSET_MAGIC, T.MSET_VERSION
    hdr["num_positions"], hdr["record_floats"] = len(positions), 5
    hdr["flags"] = flags
    hdr["model_hash"] = b"cafe"
    parts.append(hdr.tobytes())
    for game_index, turn_index, targets, *legal in positions:
        ph = np.zeros(1, dtype=T._POSITION_HEADER)
        ph["game_index"], ph["turn_index"] = game_index, turn_index
        ph["num_candidates"] = len(targets)
        ph["num_legal_moves"] = legal[0] if legal else 0
        parts.append(ph.tobytes())
        rec = np.zeros(len(targets), dtype=T._record_dtype(5))
        rec["targets"] = targets
        parts.append(rec.tobytes())
    path.write_bytes(b"".join(parts))
    path.with_suffix(".slog").touch()


def _targets(k):
    return np.tile(np.array([0.2, 0.1, 0.7, 3.0, 5.0], dtype=np.float32), (k, 1))


def test_open_leaves_corpus_sets_the_input_arm_before_the_session(tmp_path):
    """A trainer loading an open-leaves corpus must pick the opponent-leave
    input arm before the first dataset creates the FFI session, which bakes the
    arm in permanently. Reading the condition off a constructed dataset is too
    late -- it raises, and it would have built the datasets against the wrong
    row layout. Runs in a subprocess: the session is process-wide, so a test
    that creates one cannot be undone.
    """
    from scribblez.move_set_eval import targets as T

    store = tmp_path / "slogs"
    store.mkdir()
    # Two pairs: holdout_every reserves the first, so the second is the train
    # side the condition is read from.
    for stem in ("a", "b"):
        _write_mset(
            store / f"{stem}.mset",
            [(0, t, _targets(3)) for t in range(4)],
            flags=T.MSET_FLAG_OPEN_LEAVES,
        )

    code = (
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from scribblez.ffi import set_opp_leave_input\n"
        "from scribblez.move_set_eval.trainer import load_datasets\n"
        f"paths = SimpleNamespace(data_dir=Path({str(tmp_path)!r}))\n"
        "train_ds, _ = load_datasets(paths, SimpleNamespace(holdout_every=20))\n"
        "assert train_ds.open_leaves\n"
        # A no-op iff the session was created with the arm already on; the
        # guard in set_opp_leave_input raises on any other ordering.
        "set_opp_leave_input(True)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_dataset_drops_non_finite_teacher_targets(tmp_path):
    from scribblez.move_set_eval.dataset import MsetDataset

    finite = _targets(3)
    partly_bad = finite.copy()
    partly_bad[1, 4] = np.inf
    all_bad = np.full((2, 5), np.inf, dtype=np.float32)
    _write_mset(tmp_path / "a.mset", [(0, 0, finite), (0, 1, partly_bad), (0, 2, all_bad)])

    ds = MsetDataset(tmp_path)
    assert ds.num_positions == 2  # the all-bad position is gone entirely
    assert ds.num_candidates == 5  # 3 finite + 2 surviving from the partly-bad
    assert ds.dropped_candidates == 3


def test_dataset_reads_the_sweep_flag_and_its_truncation(tmp_path):
    """A swept file declares itself full-sweep and carries each position's
    legal-move count, so how much of the position the generator's cap actually
    reached is readable rather than assumed."""
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.eval import eval_slice_line
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP

    _write_mset(
        tmp_path / "sweep.mset",
        [(0, 0, _targets(4), 4), (0, 1, _targets(6), 12)],  # complete, then capped at half
        flags=MSET_FLAG_FULL_SWEEP,
    )
    ds = MsetDataset(tmp_path)
    assert ds.full_sweep
    coverage, truncated = ds.sweep_coverage
    assert coverage == pytest.approx(0.75)  # mean of 4/4 and 6/12
    assert truncated == 1
    assert "full sweep" in eval_slice_line(ds)

    # A stratified corpus records no legal counts, so nothing is claimed about
    # coverage and the eval line says the metrics are provisional.
    _write_mset(tmp_path / "strat.mset", [(0, 0, _targets(3))])
    (tmp_path / "sweep.mset").unlink()
    strat = MsetDataset(tmp_path)
    assert not strat.full_sweep
    assert strat.sweep_coverage == (1.0, 0)
    assert "provisional" in eval_slice_line(strat)


def test_dataset_refuses_to_mix_swept_and_stratified_files(tmp_path):
    """Swept positions are evaluation-only; one dataset holding both kinds is
    exactly the leak the file-level routing exists to prevent."""
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP, partition_full_sweep

    _write_mset(tmp_path / "strat.mset", [(0, 0, _targets(3))])
    _write_mset(tmp_path / "sweep.mset", [(0, 0, _targets(9), 40)], flags=MSET_FLAG_FULL_SWEEP)
    with pytest.raises(ValueError, match="mixes header flags"):
        MsetDataset(tmp_path)

    stratified, swept = partition_full_sweep(sorted(tmp_path.glob("*.mset")))
    assert [p.name for p in stratified] == ["strat.mset"]
    assert [p.name for p in swept] == ["sweep.mset"]
    assert not MsetDataset(mset_files=stratified).full_sweep
    assert MsetDataset(mset_files=swept).full_sweep


def test_batches_are_bounded_by_candidates_not_just_positions(tmp_path):
    """A swept position carries hundreds of candidates, so the position count
    alone stops bounding what a batch costs; the candidate budget takes over,
    and a position too big for it still forms a batch of its own."""
    from scribblez.move_set_eval.dataset import MsetDataset

    _write_mset(tmp_path / "a.mset", [(0, t, _targets(10), 10) for t in range(6)])
    ds = MsetDataset(tmp_path)

    batches = list(ds._batches_of(range(ds.num_positions), 100, max_candidates=25))
    assert [len(b) for b in batches] == [2, 2, 2]  # 20 candidates fits, 30 does not
    # Without a candidate budget the position bound is the only one.
    assert [len(b) for b in ds._batches_of(range(ds.num_positions), 4, None)] == [4, 2]
    # A single position over budget is not splittable, so it batches alone.
    assert [len(b) for b in ds._batches_of(range(ds.num_positions), 100, 4)] == [1] * 6


def test_regret_and_baseline_ranking_semantics():
    from scribblez.move_set_eval import eval as mset_eval

    teacher = np.array([0.5, 0.9, 0.4])  # teacher-best is index 1
    baseline = mset_eval._baseline_ranking(3)  # prefers stored order: 0, 1, 2
    assert mset_eval._topk_recall(teacher, baseline, 1) == 0.0
    assert mset_eval._topk_recall(teacher, baseline, 3) == 1.0
    assert mset_eval._regret(teacher, baseline, 1) == pytest.approx(0.4)
    assert mset_eval._regret(teacher, baseline, 2) == 0.0  # index 1 retained at k=2
    assert mset_eval._regret(teacher, teacher, 1) == 0.0  # perfect ranking forfeits nothing
    assert mset_eval._regret(teacher, baseline, 5) == 0.0  # k caps at the candidate count
