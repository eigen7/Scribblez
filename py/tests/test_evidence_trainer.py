"""Tests for the evidence trainer (roadmap items 2-3): the sim-outcome targets,
the frozen-backbone model surface (proves-best head, freeze, student init),
the trajectory dataset's rows, and -- on the GPU e2e corpus of
test_evidence_trajectories -- a training pass with the plain-vs-conditioned
metrics and the prefix-0 exactness they rest on."""

import numpy as np
import pytest
import torch
from scribblez.evidence import dataset as ED
from scribblez.evidence.train_loop import LossConfig, conditioned_forward, evaluate, run_epoch
from scribblez.move_set_eval.model import MoveSetEvalModel
from test_evidence_trajectories import traj_corpus  # noqa: F401  (fixture)
from test_move_set_eval_evidence import _synthetic_sobs


def test_sim_targets_and_gain():
    _, obs = _synthetic_sobs(4)
    obs["wins"][:] = [20, 30, 10, 30]
    obs["draws"][:] = [0, 0, 20, 0]
    obs["losses"][:] = [20, 10, 10, 10]
    wld, delta, value = ED.sim_targets(obs)
    assert np.allclose(wld.sum(axis=1), 1.0)
    assert np.allclose(value, [0.5, 0.75, 0.5, 0.75])
    assert np.allclose(delta[:, 0], 12.0) and np.allclose(delta[:, 1], 5.0)
    # Prefix 0: the gain is the value itself. Prefix 2: best-so-far is 0.75, so
    # only a candidate strictly above it gains, and the tie gains nothing.
    assert np.allclose(ED.gain_targets(value, 0), value)
    assert np.allclose(ED.gain_targets(value, 2), [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(ED.gain_targets(value, 1), [0.0, 0.25, 0.0, 0.25])


def _tiny_model() -> MoveSetEvalModel:
    torch.manual_seed(0)
    return MoveSetEvalModel(
        spatial_planes=6, scalar_size=4, trunk_channels=8, num_blocks=1, num_heads=2
    )


def test_freeze_backbone_leaves_only_the_evidence_modules_trainable():
    model = _tiny_model()
    model.freeze_backbone()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith(("evidence_fusion.", "proves_best.")) for n in trainable)
    assert {p.data_ptr() for p in model.evidence_parameters()} == {
        p.data_ptr() for n, p in model.named_parameters() if n in trainable
    }
    # The frozen trunk stays in eval mode under train(): its BatchNorm must
    # neither use batch statistics nor drift its running ones.
    model.train()
    assert not model.trunk.training and model.evidence_fusion.training


def test_load_student_tolerates_only_the_evidence_modules_missing():
    student = _tiny_model()
    state = {k: v for k, v in student.state_dict().items() if not k.startswith("proves_best.")}
    fresh = _tiny_model()
    fresh.load_student(state)  # missing proves-best head: fine
    state.pop("head.0.weight")
    with pytest.raises(ValueError, match="mismatch"):
        _tiny_model().load_student(state)


# --- e2e over the GPU trajectory corpus ---


@pytest.fixture(scope="module")
def traj_datasets(traj_corpus):  # noqa: F811
    from scribblez.ffi import set_contingent_features

    files = ED.complete_pairs(traj_corpus.dir)
    assert len(files) >= 2
    set_contingent_features(True)
    ED.adopt_information_condition(files)
    return ED.TrajectoryDataset(files[:-1]), ED.TrajectoryDataset(files[-1:])


def test_dataset_rows_follow_the_prefix(traj_datasets):
    """Every simmed candidate is a scored row; the held-out ones are those at
    or past the prefix; the gain targets are the CRN-paired improvements over
    the prefix's best; prefixes never include the uniform tail."""
    train, _ = traj_datasets
    assert train.num_positions > 0 and train.max_trajectory <= 1 + 3 + 1
    seen_prefixes = set()
    epochs = (b for e in range(4) for b in train.iter_batches(4, seed=1, epoch_index=e))
    for batch in epochs:
        pos_id = batch["move_pos_id"].numpy()
        slot = batch["slot"].numpy()
        prefixes = batch["prefix_sizes"].numpy()
        held = batch["held_out"].numpy()
        assert (held == (slot >= prefixes[pos_id])).all()
        for p, pos in enumerate(batch["positions"]):
            k = int(prefixes[p])
            seen_prefixes.add(k)
            assert k in pos.evidence_prefix_sizes()
            rows = pos_id == p
            assert rows.sum() == len(pos.moves)
            _, _, value = ED.sim_targets(pos.obs)
            assert np.allclose(batch["target_gain"].numpy()[rows], ED.gain_targets(value, k))
    assert 0 in seen_prefixes and len(seen_prefixes) > 1


def test_training_pass_moves_only_the_evidence_path(traj_datasets):
    """One pass on the frozen-backbone model: the plain outputs are unchanged
    (metrics' plain_* identical before/after), prefix-0 rows stay exact, and
    the fusion + head parameters moved."""
    train, hold = traj_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(
        spatial_planes=train.spatial_planes,
        scalar_size=train.scalar_size,
        trunk_channels=8,
        num_blocks=1,
        num_heads=2,
    ).to(device)
    model.freeze_backbone()
    before = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert before["exact_p0_maxdiff"] == 0.0
    assert before["rows"] > 0
    ev_params = [p.detach().clone() for p in model.evidence_parameters()]

    opt = torch.optim.AdamW(model.evidence_parameters(), lr=1e-2)
    cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05)
    result = run_epoch(model, opt, train.iter_batches(4, seed=0), device, cfg, max_e=8)
    assert result.rows > 0 and np.isfinite(result.losses["total"])
    after = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert after["plain_wld_ce"] == pytest.approx(before["plain_wld_ce"], abs=1e-6)
    assert after["exact_p0_maxdiff"] == 0.0
    pairs = zip(ev_params, model.evidence_parameters(), strict=True)
    assert any(not torch.equal(a, b.detach()) for a, b in pairs)
    # Conditioned outputs differ from plain on evidence-bearing rows now.
    batch = next(train.iter_batches(4, seed=3))
    plain, cond = conditioned_forward(model, batch, device, max_e=8)
    with_ev = batch["held_out"].to(device) & (
        batch["prefix_sizes"].to(device)[batch["move_pos_id"].to(device)] > 0
    )
    if bool(with_ev.any()):
        assert not torch.allclose(plain["wld"][with_ev], cond["wld"][with_ev])


# --- the trainer's store pacing (header-only .sobs stubs; no engine needed) ---


def _stub_pair(store, stem, flags=None, proposer=b"beef"):
    """A complete .slog/.sobs pair whose .sobs is a header and nothing else --
    all that file-level routing and readiness read."""
    from scribblez.sim_evidence.sobs import (
        _FILE_HEADER,
        SOBS_FLAG_TRAJECTORY,
        SOBS_MAGIC,
        SOBS_VERSION,
    )

    store.mkdir(parents=True, exist_ok=True)
    hdr = np.zeros(1, dtype=_FILE_HEADER)
    hdr["magic"], hdr["version"] = SOBS_MAGIC, SOBS_VERSION
    hdr["flags"] = SOBS_FLAG_TRAJECTORY if flags is None else flags
    hdr["proposer_hash"] = proposer
    (store / f"{stem}.sobs").write_bytes(hdr.tobytes())
    (store / f"{stem}.slog").touch()


def _params(**kw):
    from scribblez.workloads.evidence_trajectories import EvidenceTrajectoriesParams

    return EvidenceTrajectoriesParams(**kw)


def test_store_readiness_waits_for_warmup_and_a_holdout_unless_at_target(tmp_path):
    from scribblez.evidence import trainer

    store = tmp_path / "slogs"
    params = _params(warmup_pairs=3, holdout_every=4, target_pairs=0)
    assert not trainer.store_is_ready(store, params)[0]  # no store yet
    stems = [f"{1786038233456124324 + i * 4_800_000_000}-local-0" for i in range(12)]
    for s in stems[:2]:
        _stub_pair(store, s)
    ready, why = trainer.store_is_ready(store, params)
    assert not ready and "2/3" in why
    # Enough pairs, but every one hashes to the training side: still waiting.
    from scribblez.workloads import pair_store

    train, held = pair_store.split_pair_stems(stems, 4)
    for s in train[:4]:
        _stub_pair(store, s)
    ready, why = trainer.store_is_ready(store, params)
    assert not ready and "no held-out" in why
    _stub_pair(store, held[0])
    assert trainer.store_is_ready(store, params)[0]
    # A store at the target is ready regardless of either condition.
    small = tmp_path / "small"
    _stub_pair(small, stems[0])
    assert trainer.store_is_ready(small, _params(warmup_pairs=3, holdout_every=4, target_pairs=1))[
        0
    ]


def test_absorb_takes_only_new_pairs_and_keeps_sides_fixed(tmp_path, monkeypatch):
    """A pair joins the side the split assigns it the first time it is seen,
    and never again; both sides grow."""
    from scribblez.evidence import dataset as ED
    from scribblez.evidence import trainer

    class FakeDs:
        def __init__(self, files):
            self.files = list(files)
            self.absorbed = []

        def absorb(self, files):
            files = list(files)
            self.files += files
            self.absorbed.append(files)
            return len(files)

    store = tmp_path / "slogs"
    stems = [f"{1786038233456124324 + i * 4_800_000_000}-local-0" for i in range(30)]
    for s in stems[:20]:
        _stub_pair(store, s)
    train_files, hold_files = trainer.split_pairs(store, 4)
    assert train_files and hold_files
    assert set(train_files) | set(hold_files) == set(ED.complete_pairs(store))
    train_ds, hold_ds = FakeDs(train_files), FakeDs(hold_files)
    params = _params(holdout_every=4)
    assert trainer.absorb_new_pairs(store, params, train_ds, hold_ds) == 0
    for s in stems[20:]:
        _stub_pair(store, s)
    added = trainer.absorb_new_pairs(store, params, train_ds, hold_ds)
    assert added == 10
    new_train, new_hold = trainer.split_pairs(store, 4)
    assert set(train_ds.files) == set(new_train) and set(hold_ds.files) == set(new_hold)
    assert not (set(train_ds.files) & set(hold_ds.files))
    assert trainer.absorb_new_pairs(store, params, train_ds, hold_ds) == 0


def test_dataset_refuses_a_mixed_corpus(tmp_path):
    from scribblez.evidence.dataset import TrajectoryDataset

    store = tmp_path / "slogs"
    _stub_pair(store, "a", proposer=b"beef")
    _stub_pair(store, "b", proposer=b"cafe")
    _stub_pair(store, "c", flags=0)  # not a trajectory file
    with pytest.raises(ValueError, match="mixes proposers"):
        TrajectoryDataset([store / "a.sobs", store / "b.sobs"])
    with pytest.raises(ValueError, match="not a trajectory"):
        TrajectoryDataset([store / "c.sobs"])


# --- the runner, end to end on the GPU corpus ---


def _student_checkpoint(path, train, *, open_leaves: bool, version=None):
    """A move_set_eval-style rolling checkpoint (weights + config) of a tiny
    student matching the corpus's input widths."""
    from scribblez.move_set_eval.moves import move_encoding_version

    model = MoveSetEvalModel(
        spatial_planes=train.spatial_planes,
        scalar_size=train.scalar_size,
        trunk_channels=8,
        num_blocks=1,
        num_heads=2,
    )
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("proves_best.")}
    torch.save(
        {
            "model_state_dict": state,
            "config": {
                "spatial_planes": train.spatial_planes,
                "scalar_size": train.scalar_size,
                "trunk_channels": 8,
                "num_blocks": 1,
                "num_heads": 2,
                "contingent_features": True,
                "open_leaves": open_leaves,
                "move_encoding_version": version
                if version is not None
                else move_encoding_version(),
            },
        },
        path,
    )


class _Sink:
    kind = "local"

    def push_json(self, rel, obj):
        pass


def _ctx(tmp_path, tag, params):
    from types import SimpleNamespace

    from scribblez import workloads
    from scribblez.workloads.base import WorkerContext

    spec = workloads.get("evidence_trajectories")
    ctx = WorkerContext(
        spec=spec,
        role=spec.role("train"),
        tag=tag,
        params=params,
        worker_id="local-0",
        threads=2,
        max_cycles=0,
        sink=_Sink(),
    )
    ctx.tag_paths = lambda: spec.paths(tag, tmp_path)  # a scratch mount root
    return SimpleNamespace(ctx=ctx, paths=spec.paths(tag, tmp_path))


def test_batched_evidence_builder_matches_the_per_position_one(traj_datasets):
    from scribblez.evidence.train_loop import _INPUT_KEYS, _MOVE_KEYS, batch_evidence_inputs
    from scribblez.move_set_eval.evidence import build_evidence_inputs, collate_evidence

    train, _ = traj_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(train.spatial_planes, train.scalar_size, 8, 1, 2).to(device).eval()
    max_e = 8
    for batch in train.iter_batches(4, seed=2):
        move_args = tuple(batch[k].to(device) for k in _MOVE_KEYS)
        spatial, scalar = (batch[k].to(device) for k in _INPUT_KEYS)
        pos_id, slot = move_args[-1], batch["slot"].to(device)
        with torch.no_grad():
            board, g = model.encode_board(spatial, scalar)
            plain = model.score_moves(board, g, model.encode_moves(board, *move_args), pos_id)
            fast = batch_evidence_inputs(batch, move_args, plain, max_e, device)
            items = []
            for p, pos in enumerate(batch["positions"]):
                k = int(batch["prefix_sizes"][p])
                rows = (pos_id == p) & (slot < k)
                first = {kk: plain[kk][rows] for kk in ("wld", "score_diff", "planes")}
                items.append(
                    build_evidence_inputs(
                        pos.moves[:k],
                        pos.obs[:k],
                        int(batch["pre_move_diff"][p]),
                        first,
                        max_e=max_e,
                        device=device,
                    )
                )
            slow = collate_evidence(items)
        import dataclasses

        for f in dataclasses.fields(fast):
            a, b = getattr(fast, f.name), getattr(slow, f.name)
            assert a.shape == b.shape
            if a.dtype == torch.bool:
                assert torch.equal(a, b)
            else:
                assert torch.allclose(a.float(), b.float(), atol=1e-6)


def test_run_trains_to_its_budget_resumes_and_refuses_mismatches(
    traj_corpus,  # noqa: F811
    traj_datasets,
    tmp_path,
):
    """The train role end to end on the GPU corpus: refuses a student on the
    wrong information-condition arm or move-encoding version, then trains its
    epoch budget, writes per-pass checkpoints and metrics, and a second
    invocation resumes and stops at once."""
    from scribblez.evidence import dataset as ED
    from scribblez.evidence import trainer

    train, _ = traj_datasets
    store_src = traj_corpus.dir
    tag = "zz-evidence-run"
    ckpt = tmp_path / "student.pt"
    _student_checkpoint(ckpt, train, open_leaves=False)
    base = dict(
        proposer_model="/x",
        teacher_model="/x",
        student_checkpoint=str(ckpt),
        warmup_pairs=1,
        holdout_every=2,
        batch_positions=4,
        target_pairs=0,
    )
    params = _params(train_epochs=2, **base)
    scratch = _ctx(tmp_path, tag, params)
    store = scratch.paths.data_dir / "slogs"
    store.mkdir(parents=True)
    import os
    import shutil
    import time

    for f in ED.complete_pairs(store_src):
        shutil.copy(f, store / f.name)
        shutil.copy(f.with_suffix(".slog"), store / f.with_suffix(".slog").name)
    # Make the store final so passes spend the budget: with no declared size
    # the clock reads a store older than QUIET_SECONDS as done.
    stale = time.time() - 3600
    for f in store.iterdir():
        os.utime(f, (stale, stale))

    # Refusals come before any training.
    bad = tmp_path / "bad_arm.pt"
    _student_checkpoint(bad, train, open_leaves=True)
    assert (
        trainer.run(
            _ctx(
                tmp_path, tag, _params(train_epochs=2, **{**base, "student_checkpoint": str(bad)})
            ).ctx
        )
        == 1
    )
    bad_v = tmp_path / "bad_version.pt"
    _student_checkpoint(bad_v, train, open_leaves=False, version=999)
    assert (
        trainer.run(
            _ctx(
                tmp_path, tag, _params(train_epochs=2, **{**base, "student_checkpoint": str(bad_v)})
            ).ctx
        )
        == 1
    )
    assert not scratch.paths.checkpoints_dir.exists()

    assert trainer.run(scratch.ctx) == 0
    assert (scratch.paths.checkpoints_dir / "model_epoch_0000.pt").exists()
    assert (scratch.paths.checkpoints_dir / "model_epoch_0001.pt").exists()
    import sqlite3

    conn = sqlite3.connect(scratch.paths.dashboard_db)
    epochs = {r[0] for r in conn.execute("select epoch from metrics")}
    names = {r[0] for r in conn.execute("select name from metrics")}
    assert epochs == {0, 1}
    assert {"loss", "cond_wld_ce", "plain_wld_ce", "gain_hit_acc", "exact_p0_maxdiff"} <= names
    # Resume: the budget is spent, so the second run stops without a pass.
    assert trainer.run(scratch.ctx) == 0
    assert not (scratch.paths.checkpoints_dir / "model_epoch_0002.pt").exists()
