"""Tests for the move set evaluation model training side: the move-feature encoder
(pure CPU) and the end-to-end dataset -> train step -> recall eval path over a
tiny generated .mset/.slog corpus (GPU, since the target generator builds a
TensorRT engine).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.move_set_eval import moves as move_enc
from scribblez.sim_evidence.sobs import MOVE_DTYPE, MOVE_PLAY, move_footprint

# This checkout's own binaries, so a worktree's tests exercise the code built
# beside them rather than the primary checkout's (as in test_move_set_eval_targets.py).
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

# The same principle for subprocesses running a script file: their sys.path[0]
# is the script's directory, so without this override the image-wide
# PYTHONPATH=/workspace/repo/py would hand a worktree's subprocess the primary
# checkout's scribblez.
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
    ),
}
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


def test_schedule_free_recalibrates_the_mset_trunk_batchnorm():
    """The schedule-free bracket on the real mset model, CPU-only: one step
    trains the base iterate, eval_mode deploys the different averaged one, and
    recalibrates the board trunk's BatchNorm for it through encode_board -- the
    forward hook the trainer passes, since the model's BatchNorm is all in the
    trunk and its plain forward would demand the candidate set."""
    from types import SimpleNamespace

    from scribblez.generational.optim import ScheduleFreeArm, build_optimizer
    from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE
    from scribblez.move_set_eval.model import MoveSetEvalModel
    from scribblez.move_set_eval.trainer import _encode_board_only

    batch = _ragged_batch([3, 5, 2, 8])
    torch.manual_seed(0)
    model = MoveSetEvalModel(
        spatial_planes=batch["input_spatial"].shape[1],
        scalar_size=batch["input_scalar"].shape[1],
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    params = SimpleNamespace(
        optimizer=OPTIMIZER_SCHEDULE_FREE, lr=1e-3, weight_decay=1e-4, lr_warmup_rows=0
    )
    opt = build_optimizer(model, params, rows_per_step=64)
    arm = ScheduleFreeArm(params, opt)

    # float32 board + move inputs (the dataset emits float64 for tight equality
    # checks elsewhere; the model runs in its default float32 here).
    fwd = [
        batch["input_spatial"].float(),
        batch["input_scalar"].float(),
        batch["move_letters"],
        batch["move_blanks"],
        batch["move_squares"],
        batch["move_tile_mask"].float(),
        batch["move_scalars"].float(),
        batch["move_pos_id"],
    ]
    first_bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))

    # A few steps: the averaged iterate is the single base iterate after one
    # step, so it only diverges from the training weights once there are several.
    arm.train_mode()
    for _ in range(3):
        opt.zero_grad()
        out = model(*fwd)
        sum(v.sum() for v in out.values()).backward()
        opt.step()
    trained = [p.detach().clone() for p in model.parameters()]
    bn_at_training = first_bn.running_mean.clone()

    board_batch = [(fwd[0], fwd[1])]
    arm.eval_mode(model, board_batch, forward_fn=_encode_board_only)

    # The deployed iterate differs from the trained one (the swap did something)
    # and the trunk's running statistics were recomputed for it.
    assert not all(torch.equal(a, b) for a, b in zip(trained, model.parameters(), strict=True))
    assert not torch.equal(bn_at_training, first_bn.running_mean)


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
    assert not ds.has_planes  # the evaluation slice is plane-less by design
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
    assert "plane_bce" not in metrics  # no plane targets on this slice
    for k in (1, 3, 5):
        assert 0.0 <= metrics[f"recall@{k}"] <= 1.0
        assert metrics[f"regret@{k}"] >= 0.0
    # The baseline over a sweep is the exact static-equity ranking, which is a
    # real move ordering rather than a shuffle: it must beat a coin flip.
    assert metrics["spearman_baseline"] > 0.0
    # Exchange-slice metrics (the A4 dedicated-head readout): a sweep keeps
    # every exchange candidate, so eligible positions exist in any corpus with
    # bag >= 7 turns, and both metrics stay in range.
    assert metrics["positions_with_exchanges"] > 0
    for suffix in ("", "_baseline"):
        assert metrics[f"exch_rank_regret{suffix}"] >= 0.0
        for k in (1, 3, 5):
            assert 0.0 <= metrics[f"exch_retention@{k}{suffix}"] <= 1.0


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
        # A stratified corpus carries the teacher's placement planes,
        # dequantized to per-cell probabilities.
        assert ds.has_planes
        assert batch["target_planes"].shape == (m, 4, 225)
        assert batch["target_planes"].dtype == torch.float32
        assert float(batch["target_planes"].min()) >= 0.0
        assert float(batch["target_planes"].max()) <= 1.0
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
    loss_cfg = LossConfig(
        lambda_sd=0.004, huber_delta_mean=10.0, huber_delta_std=10.0, lambda_planes=1.0
    )

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
        # The plane head trains against this corpus's plane targets.
        assert np.isfinite(result.losses["planes"]) and result.losses["planes"] > 0.0
        first = result.losses["total"] if first is None else first
        last = result.losses["total"]
    assert rows == 3 * ds.num_candidates
    assert last <= first * 2.0  # a few steps should not diverge

    metrics = evaluate(model, ds, device, positions_per_batch=8)
    assert metrics["positions"] == ds.num_positions
    # This slice carries plane targets, so the plane-readout metric is
    # reported (and BCE is positive for any non-degenerate model).
    assert metrics["plane_bce"] > 0.0
    for k in (1, 3, 5):
        for suffix in ("", "_baseline"):
            assert 0.0 <= metrics[f"recall@{k}{suffix}"] <= 1.0
            assert metrics[f"regret@{k}{suffix}"] >= 0.0
    # Larger K keeps a superset of the candidates, so regret can only shrink.
    assert metrics["regret@5"] <= metrics["regret@3"] <= metrics["regret@1"]
    for suffix in ("", "_baseline"):
        assert -1.0 <= metrics[f"spearman{suffix}"] <= 1.0
    assert metrics["positions_with_exchanges"] >= 0  # stratified: may be sparse


def test_move_encoding_version_is_the_engines():
    from scribblez.move_set_eval.moves import move_encoding_version

    # v1 = exchanges carry their surrendered tiles (move_set_encoder.h). A bump
    # without a coordinated retrain story should fail loudly here.
    assert move_encoding_version() == 1


def test_publish_config_records_params_before_the_model_exists(tmp_path):
    """The Info tab's params are written up front (parameter count re-stamped
    once the model is built), so the dashboard shows the run's config while the
    trainer is still filling the store toward warmup_pairs -- as position_eval
    does -- rather than staying blank until training starts."""
    import json

    from scribblez.dashboard import db
    from scribblez.move_set_eval import trainer
    from scribblez.workloads.move_set_eval import MoveSetEvalParams

    conn = db.connect(tmp_path / "dashboard.db")
    params = MoveSetEvalParams(teacher_tag="teach", teacher_generation=3, lambda_sd=0.01)
    trainer.publish_config(conn, "tag1", params)

    meta = db.read_meta(conn)
    assert meta is not None  # the row exists before any model or training pass
    args = json.loads(meta["args_json"])
    assert args["teacher_tag"] == "teach" and args["teacher_generation"] == 3
    assert meta["model_params"] == 0  # unknown until the model is built and re-stamps it


def _write_mset(path, positions, flags=0):
    """Hand-pack a minimal plane-less .mset (layout mirrored by targets.py's
    dtypes), plus its companion .slog. A position is (game_index, turn_index,
    targets), optionally followed by the swept position's legal-move count."""
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
        rec = np.zeros(len(targets), dtype=T._record_dtype(5, 0))
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


def _pair(store, stem, flags=0, positions=4):
    _write_mset(
        store / f"{stem}.mset", [(0, t, _targets(3)) for t in range(positions)], flags=flags
    )


def _params(**kw):
    from scribblez.workloads.move_set_eval import MoveSetEvalParams

    return MoveSetEvalParams(**kw)


def test_training_waits_for_a_corpus_worth_starting_on(tmp_path):
    """A trainer started beside its generator must not lock the run into the
    handful of pairs that exist in the first seconds -- neither the tiny
    training corpus nor, worse, a holdout that is empty (aliased to the
    training set for the whole run) or that is the stratified fallback frozen
    in before the first swept pair arrives."""
    from scribblez.move_set_eval import trainer
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP

    store = tmp_path / "slogs"
    store.mkdir()
    params = _params(warmup_pairs=3, sweep_every=20)

    assert trainer.store_is_ready(store, params) == (False, "0/3 pairs")
    for i in range(3):
        _pair(store, f"s{i}")
    # Enough pairs, but nothing to read the A3 gate metrics on yet.
    ready, why = trainer.store_is_ready(store, params)
    assert not ready and "no held-out pair yet" in why

    _pair(store, "sweep0", flags=MSET_FLAG_FULL_SWEEP)
    assert trainer.store_is_ready(store, params) == (True, "")

    # A tag that asks for no holdout at all waits only for the pair count.
    assert trainer.store_is_ready(store, _params(warmup_pairs=3, sweep_every=0, holdout_every=0))[0]


def test_an_empty_store_is_never_ready_however_low_the_warmup(tmp_path):
    """warmup_pairs=0 asks for no warmup, not for training on nothing: without
    a floor the trainer starts and dies on load_datasets' FileNotFoundError."""
    from scribblez.move_set_eval import trainer

    store = tmp_path / "slogs"
    store.mkdir()
    assert not trainer.store_is_ready(store, _params(warmup_pairs=0, sweep_every=0))[0]


def test_corpus_clock_waits_for_the_target_and_for_the_stragglers(tmp_path):
    """The trainer's signal that its corpus is done. Reaching the declared size
    is not enough on its own: a second generate worker that was mid-cycle when
    the first crossed the target still has pairs to land, and epochs counted
    before they do would be scored against a different holdout than the ones
    after."""
    from scribblez.move_set_eval import trainer

    store = tmp_path / "slogs"
    store.mkdir()
    for i in range(2):
        _pair(store, f"s{i}")
    clock = trainer.corpus_clock(store, _params(target_pairs=3))

    assert not clock.is_final(absorbed=0)  # short of the target
    _pair(store, "s2")
    assert not clock.is_final(absorbed=40)  # at the target, but pairs still landing
    assert clock.is_final(absorbed=0)  # quiet at the target


def test_corpus_clock_reads_staleness_off_the_store_not_its_own_history(tmp_path):
    """With no declared size, whether the corpus is done is a property of the
    store: one written to moments ago has a generator behind it, one whose
    newest pair is old does not. Judging it from what this worker has watched
    instead would call a live store static on a fresh start and again after
    every restart -- neither has seen the generator from the beginning."""
    import os

    from scribblez.move_set_eval import trainer
    from scribblez.workloads import pair_store

    store = tmp_path / "slogs"
    store.mkdir()
    _pair(store, "s0")
    params = _params(target_pairs=0)

    # A store just delivered into, seen by a worker that has watched nothing.
    assert not trainer.corpus_clock(store, params).is_final(absorbed=0)

    stale = time.time() - pair_store.QUIET_SECONDS - 1
    for f in store.iterdir():
        os.utime(f, (stale, stale))
    # Same worker, same absence of history: now the store itself says it is done.
    assert trainer.corpus_clock(store, params).is_final(absorbed=0)

    _pair(store, "s1")  # the generator was not gone after all
    assert not trainer.corpus_clock(store, params).is_final(absorbed=0)


def test_absorb_adds_only_the_new_files(tmp_path):
    """Keeping up with a running generator costs the new pairs, not a re-read
    of the corpus, and the two sides of the split track their own files."""
    from scribblez.move_set_eval.dataset import MsetDataset

    store = tmp_path / "slogs"
    store.mkdir()
    _pair(store, "a", positions=4)
    ds = MsetDataset(mset_files=[store / "a.mset"])
    assert ds.num_positions == 4
    assert [p.name for p in ds.files] == ["a.mset"]

    _pair(store, "b", positions=6)
    assert ds.absorb([store / "b.mset"]) == 6
    assert ds.num_positions == 10
    assert [p.name for p in ds.files] == ["a.mset", "b.mset"]
    assert ds.absorb([]) == 0  # a quiet pass changes nothing

    # Batches address the absorbed file by its own replay, not the first one's.
    assert {p.file_id for p in ds._positions} == {0, 1}


def test_absorb_refuses_a_file_from_another_corpus(tmp_path):
    """The mixed-teacher / mixed-condition guard has to hold for files arriving
    later, not just the ones present at construction."""
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP

    store = tmp_path / "slogs"
    store.mkdir()
    _pair(store, "a")
    _pair(store, "swept", flags=MSET_FLAG_FULL_SWEEP)
    ds = MsetDataset(mset_files=[store / "a.mset"])
    with pytest.raises(ValueError, match="mixes header flags"):
        ds.absorb([store / "swept.mset"])


def test_a_run_started_beside_its_generator_trains_on_the_whole_corpus(tmp_path):
    """The whole point, end to end: a trainer started when the store held a
    fraction of the corpus must finish having trained on all of it, spending
    its epoch budget only once generation is done.

    This drives run()'s loop verbatim with the training call removed -- the
    wait, the per-pass absorb, the budget rule -- against a store that grows
    underneath it, as the generate worker grows the real one.
    """
    from types import SimpleNamespace

    from scribblez.move_set_eval import trainer
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP

    store = tmp_path / "slogs"
    store.mkdir()
    paths = SimpleNamespace(data_dir=tmp_path)
    params = _params(warmup_pairs=2, target_pairs=8, train_epochs=3, sweep_every=20)

    delivered = []

    def deliver_one():
        """One generator cycle: every 3rd pair swept, as sweep_every does."""
        i = len(delivered)
        swept = i % 3 == 2
        _pair(store, f"p{i:02d}", flags=MSET_FLAG_FULL_SWEEP if swept else 0, positions=4)
        delivered.append(swept)

    for _ in range(3):  # the store as it stands when the trainer is started
        deliver_one()
    assert trainer.store_is_ready(store, params)[0]

    train_ds, holdout_ds = trainer.load_datasets(paths, params)
    started_with = train_ds.num_positions
    state = trainer.MsetTrainState()
    clock = trainer.corpus_clock(store, params)

    passes = 0
    while trainer.epochs_left(params, state):
        absorbed = trainer.absorb_new_pairs(paths, params, train_ds, holdout_ds)
        settled = clock.is_final(absorbed)
        state.settled_epochs += int(settled)
        state.generation_index += 1
        passes += 1
        assert passes < 50, "loop failed to converge"
        if len(delivered) < params.target_pairs:
            deliver_one()  # the generator is still running

    swept_pairs = sum(delivered)
    assert state.settled_epochs == 3  # the budget bought passes over the finished corpus
    assert state.generation_index > 3  # and the growth passes were free
    assert train_ds.num_positions == (len(delivered) - swept_pairs) * 4
    assert holdout_ds.num_positions == swept_pairs * 4
    assert train_ds.num_positions > started_with  # it did not train the snapshot it started on


def test_a_small_generation_target_releases_the_wait_rather_than_hanging(tmp_path):
    """A target below warmup_pairs, or too small to have drawn a swept pair,
    would otherwise leave the trainer polling for pairs no one will ever
    deliver -- a tag that can never finish, with both workers looking alive."""
    from scribblez.move_set_eval import trainer

    store = tmp_path / "slogs"
    store.mkdir()
    params = _params(warmup_pairs=100, target_pairs=3, sweep_every=20)

    for i in range(2):
        _pair(store, f"s{i}")
    assert not trainer.store_is_ready(store, params)[0]  # still short of the target

    _pair(store, "s2")  # target reached, though warmup and the sweep gate are not
    assert trainer.store_is_ready(store, params) == (True, "")


# Drives trainer.run() in its own process: run() fixes the FFI session's
# feature arm, which is process-wide and already set by the time any other test
# in this file has touched the dataset.
_DRIVE_RUN = """
import sys, torch
import onnx
from types import SimpleNamespace
from pathlib import Path
from scribblez.move_set_eval import trainer
from scribblez.workloads.move_set_eval import MoveSetEvalParams

root = Path(sys.argv[1])
pairs = len(list((root / "slogs").glob("*.mset")))
params = MoveSetEvalParams(
    warmup_pairs=1,
    target_pairs=pairs,   # already at size, so every pass is over a final corpus
    train_epochs=2,
    batch_positions=8,
    trunk_channels=8,
    num_blocks=2,
    num_heads=2,
)
paths = SimpleNamespace(
    root=root,
    data_dir=root,
    dashboard_db=root / "dashboard.db",
    rolling_checkpoint=root / "checkpoints" / "model.pt",
    onnx_path=lambda epoch: root / "models" / f"model_epoch_{epoch:04d}.onnx",
)
ctx = SimpleNamespace(
    params=params, tag="t", worker_id="w0", threads=1, kind="local",
    sink=SimpleNamespace(
        kind="local", push_json=lambda *a, **k: None, read_json=lambda *a, **k: None
    ),
    role=SimpleNamespace(name="train"), provenance={},
    tag_paths=lambda: paths,
)
assert trainer.run(ctx) == 0, "run() did not exit cleanly"

saved = torch.load(paths.rolling_checkpoint, map_location="cpu", weights_only=False)
assert saved["settled_epochs"] == 2, saved["settled_epochs"]
assert saved["generation_index"] == 2, saved["generation_index"]
assert saved["rows_trained"] > 0, saved["rows_trained"]
# The per-pass ONNX exports landed beside the checkpoint, metadata stamped
# from the run's own config (the self-describing fields).
for epoch in (0, 1):
    meta = {e.key: e.value for e in onnx.load(str(paths.onnx_path(epoch))).metadata_props}
    assert meta["graph"] == "move_set_eval", meta
    assert meta["move_encoding_version"] == "1", meta
    assert meta["opp_leave_input"] == "false", meta
print("OK")
"""


def test_run_stops_after_its_budget_over_the_finished_corpus(corpus_dir, sweep_dir, tmp_path):
    """trainer.run() itself, end to end on a real corpus: it must spend its
    budget on passes over the finished store and then stop.

    The hand-driven test above exercises the helpers; only calling run() proves
    the loop spends the budget where the trainer actually spends it.
    """
    # The sweep fixture relabels the SAME .slog stems, so the swept pairs are
    # copied under their own stems rather than overwriting the stratified ones.
    store = tmp_path / "slogs"
    store.mkdir()
    for src in corpus_dir.glob("*.mset"):
        for member in (src, src.with_suffix(".slog")):
            shutil.copy(member, store / member.name)
    for src in sweep_dir.glob("*.mset"):
        for member in (src, src.with_suffix(".slog")):
            shutil.copy(member, store / f"sweep-{member.name}")
    assert len(list(store.glob("*.mset"))) > len(list(corpus_dir.glob("*.mset")))

    script = tmp_path / "drive_run.py"
    script.write_text(_DRIVE_RUN)
    # Bounded: a budget that never advances makes run() loop forever, which is
    # the production symptom -- it has to surface as a failure, not a hang.
    try:
        out = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=_SUBPROCESS_ENV,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("trainer.run() did not stop: the epoch budget is not being spent")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "OK" in out.stdout


def test_absorb_grows_a_shared_holdout_once(tmp_path):
    """With no held-out pairs the metrics run on the training set, and the two
    dataset handles are the same object -- which must be ingested into once,
    not twice, or every later pair is counted double."""
    from types import SimpleNamespace

    from scribblez.move_set_eval import trainer

    store = tmp_path / "slogs"
    store.mkdir()
    paths = SimpleNamespace(data_dir=tmp_path)
    params = _params(sweep_every=0, holdout_every=0)  # no holdout at all
    _pair(store, "a", positions=4)

    train_ds, holdout_ds = trainer.load_datasets(paths, params)
    assert holdout_ds is train_ds

    _pair(store, "b", positions=6)
    assert trainer.absorb_new_pairs(paths, params, train_ds, holdout_ds) == 6
    assert train_ds.num_positions == 10  # not 16


def test_a_late_swept_pair_cannot_reshuffle_the_sides_or_crash_the_run(tmp_path):
    """A run that started before any swept pair existed uses the stratified
    fallback split. The first swept pair to land turns EVERY stratified pair
    into a training pair, so a trainer that simply re-took the split would hand
    its holdout's files to the training set -- and offer a swept file to a
    stratified dataset, which refuses mixed headers and would kill the worker.
    """
    from types import SimpleNamespace

    from scribblez.move_set_eval import trainer
    from scribblez.move_set_eval.targets import MSET_FLAG_FULL_SWEEP

    store = tmp_path / "slogs"
    store.mkdir()
    paths = SimpleNamespace(data_dir=tmp_path)
    params = _params(sweep_every=20, holdout_every=2)
    for i in range(6):
        _pair(store, f"p{i}", positions=4)

    train_ds, holdout_ds = trainer.load_datasets(paths, params)
    assert holdout_ds is not train_ds
    held_before = set(holdout_ds.files)
    trained_before = set(train_ds.files)
    assert held_before and trained_before

    _pair(store, "zz", flags=MSET_FLAG_FULL_SWEEP, positions=4)  # the straggler
    # Must not raise, whatever the recomputed split now says.
    trainer.absorb_new_pairs(paths, params, train_ds, holdout_ds)

    assert set(holdout_ds.files) == held_before  # nothing left the holdout
    assert set(train_ds.files) == trained_before  # and nothing joined training
    assert not set(train_ds.files) & set(holdout_ds.files)  # no pair on both sides
