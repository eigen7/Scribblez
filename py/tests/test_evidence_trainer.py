"""Tests for the evidence trainer (roadmap items 2-3, 5): the sim-outcome
targets, the subset-assembly dataset (arbitrary evidence subsets, the single
membership tensor threading its five seams, the batched evidence builder), the
frozen-backbone model surface (proves-best head, freeze, student init), and --
on the GPU e2e corpus of test_evidence_trajectories -- a training pass with the
plain-vs-conditioned metrics and the empty-subset exactness they rest on, in
both the frozen mode and the unfrozen (joint distillation + sim-outcome) one."""

import numpy as np
import pytest
import torch
from scribblez.evidence import dataset as ED
from scribblez.evidence.train_loop import (
    Distillation,
    LossConfig,
    batch_evidence_inputs,
    conditioned_forward,
    evaluate,
    run_epoch,
)
from scribblez.footprint_spatial import NUM_CLASSES
from scribblez.move_set_eval import moves as move_enc
from scribblez.move_set_eval import train_loop as mset_train_loop
from scribblez.move_set_eval.dataset import MsetDataset
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.sim_evidence.sobs import (
    ROLE_ANCHOR,
    ROLE_OFF_POLICY,
    ROLE_ON_POLICY,
    SobsPosition,
)
from test_evidence_trajectories import traj_corpus  # noqa: F401  (fixture)
from test_move_set_eval_evidence import _synthetic_sobs


def test_packed_obs_round_trips_records_exactly():
    """The dataset's sparse retention (_PackedObs) must rebuild the verbatim
    .sobs records: retaining the near-empty 2927-wide histograms dense is what
    would blow trainer RAM at corpus scale, and any lossy repack would corrupt
    the observed half of every evidence token."""
    _, obs = _synthetic_sobs(3)
    rng = np.random.default_rng(0)
    for name in ("opp_next_count", "self_next_count"):
        obs[name][:] = 0
        obs[name][rng.random(obs[name].shape) < 0.02] = 7
    obs["opp_win_count"][:] = obs["opp_next_count"] * 0.5
    obs["self_win_count"][:] = 0.0
    dense = ED._PackedObs(obs).densify()
    assert dense.dtype == obs.dtype
    assert dense.tobytes() == obs.tobytes()


def test_dataset_retains_obs_sparse_but_batches_carry_dense_records(traj_datasets):
    """_TrajPosition empties its SobsPosition's dense records (the RAM guard),
    while every batch's `positions` re-attach obs rows identical to `all_obs`.
    The retained moves/roles must OWN their data: as read_sobs field views
    their .base would silently pin the whole dense record buffer."""
    train, _ = traj_datasets
    for pos in train._positions:
        assert len(pos.sobs.obs) == 0
        assert pos.sobs.moves.base is None
        assert pos.sobs.roles.base is None
    batch = next(train.iter_batches(4, seed=0))
    offset = 0
    for pos in batch["positions"]:
        k = len(pos.moves)
        assert pos.obs.tobytes() == batch["all_obs"][offset : offset + k].tobytes()
        offset += k


def test_sim_targets_and_gain():
    _, obs = _synthetic_sobs(4)
    obs["wins"][:] = [20, 30, 10, 30]
    obs["draws"][:] = [0, 0, 20, 0]
    obs["losses"][:] = [20, 10, 10, 10]
    wld, delta, value = ED.sim_targets(obs)
    assert np.allclose(wld.sum(axis=1), 1.0)
    assert np.allclose(value, [0.5, 0.75, 0.5, 0.75])
    assert np.allclose(delta[:, 0], 12.0) and np.allclose(delta[:, 1], 5.0)

    def subset(*members):
        m = np.zeros(4, dtype=bool)
        m[list(members)] = True
        return m

    # Empty subset: the gain is the value itself. best-so-far is a max over the
    # subset's members, not a leading prefix -- a subset {0,3} (values 0.5, 0.75)
    # tops out at 0.75, so only a candidate strictly above it gains and the tie
    # gains nothing; a non-contiguous {1} (0.75) baselines the same at 0.75.
    assert np.allclose(ED.gain_targets(value, subset()), value)
    assert np.allclose(ED.gain_targets(value, subset(0, 3)), [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(ED.gain_targets(value, subset(0)), [0.0, 0.25, 0.0, 0.25])
    assert np.allclose(ED.gain_targets(value, subset(1)), [0.0, 0.0, 0.0, 0.0])


# --- subset assembly (CPU, synthetic v4 positions -- no GPU, no model) ---


def _traj_pos(roles, num_legal_moves: int = 64) -> SobsPosition:
    """A synthetic trajectory position carrying real (distinct-per-candidate)
    move and observation records under the given per-candidate roles."""
    roles = np.asarray(roles, dtype=np.uint8)
    moves, obs = _synthetic_sobs(len(roles))
    return SobsPosition(0, 0, 100, 0, num_legal_moves, 0, moves, obs, roles)


_ROLES = [ROLE_ANCHOR, ROLE_ON_POLICY, ROLE_ON_POLICY, ROLE_ON_POLICY, ROLE_OFF_POLICY]


def test_assemble_subset_holds_only_the_anchor_and_on_policy_within_the_cap():
    """A drawn subset is empty or contains the anchor; it never holds an
    off-policy (labels-only) draw; and it fits min(num_evidence, cap)."""
    pos = _traj_pos(_ROLES)
    assert pos.num_evidence == 4  # anchor + 3 on-policy
    rng = np.random.default_rng(0)
    seen_sizes, seen_members = set(), set()
    for _ in range(400):
        mask = ED.assemble_subset(rng, pos, max_evidence_width=3)
        size = int(mask.sum())
        seen_sizes.add(size)
        assert size <= 3  # capped below num_evidence
        assert not (mask & (pos.roles == ROLE_OFF_POLICY)).any()  # never labels-only
        if size:
            assert mask[0]  # the anchor is in every non-empty subset
            seen_members.update(np.flatnonzero(mask).tolist())
    assert seen_sizes == {0, 1, 2, 3}
    # Non-contiguous subsets occur: on-policy slots 1..3 all appear as members.
    assert {1, 2, 3} <= seen_members


def test_assemble_subset_empty_fraction_pins_the_zero_prefix_rate():
    """empty_fraction fixes P(empty subset) -- the all-held-out rows that set
    the rows-clocked LR horizon -- and the default sweeps size uniformly."""
    pos = _traj_pos(_ROLES)
    rng = np.random.default_rng(1)
    assert all(not ED.assemble_subset(rng, pos, empty_fraction=1.0).any() for _ in range(20))
    assert all(ED.assemble_subset(rng, pos, empty_fraction=0.0).any() for _ in range(20))
    empties = sum(not ED.assemble_subset(rng, pos, empty_fraction=0.25).any() for _ in range(2000))
    assert 0.2 < empties / 2000 < 0.3  # ~1/4
    # Default: uniform over {0..num_evidence}, so empties at ~1/(4+1).
    default = sum(not ED.assemble_subset(rng, pos).any() for _ in range(2000))
    assert 0.15 < default / 2000 < 0.25


def test_compact_index_packs_members_in_slot_order():
    """_compact_index gives each member its 0-based rank in slot order -- the
    padded slot an arbitrary subset scatters to, the way the deployment builder
    packs {0, 2, 3} into rows 0, 1, 2."""
    mask = np.array([True, False, True, True, False])
    assert ED._compact_index(mask).tolist() == [0, 0, 1, 2, 0]
    assert ED._compact_index(np.zeros(3, bool)).tolist() == [0, 0, 0]


def _cpu_evidence_batch(units, pre_diffs):
    """The subset of a batch dict + move_args batch_evidence_inputs reads, built
    from (position, subset mask) units the way _build_batch does (real
    _compact_index), so the seam is exercised without board inputs or a GPU."""
    positions = [pos for pos, _ in units]
    masks = [mask for _, mask in units]
    all_moves = np.concatenate([pos.moves for pos in positions])
    all_obs = np.concatenate([pos.obs for pos in positions])
    counts = [len(pos.moves) for pos in positions]
    pos_id = np.repeat(np.arange(len(units), dtype=np.int64), counts)
    in_evidence = np.concatenate(masks)
    ev_index = np.concatenate([ED._compact_index(mask) for mask in masks])
    enc = move_enc.encode_moves(all_moves, np.asarray(pre_diffs, np.int32)[pos_id])
    move_keys = ("letters", "blanks", "squares", "tile_mask", "scalars")
    move_args = (*(torch.from_numpy(enc[k]) for k in move_keys), torch.from_numpy(pos_id))
    batch = {
        "positions": positions,
        "in_evidence": torch.from_numpy(in_evidence),
        "ev_index": torch.from_numpy(ev_index),
        "all_moves": all_moves,
        "all_obs": all_obs,
    }
    return batch, move_args


def _fabricated_plain(m: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "wld": torch.randn(m, 3, generator=gen),
        "score_diff": torch.randn(m, 2, generator=gen),
        "planes": torch.randn(m, 4, NUM_CLASSES, generator=gen),
    }


def test_batch_evidence_inputs_aligns_the_halves_for_arbitrary_subsets():
    """The batched builder over non-contiguous subsets equals collating the
    deployment per-position builder over each subset's members: the observed
    half (raw .sobs records) and the predicted half (the plain pass) are
    gathered in the same enumeration, so a token's observation and prediction
    describe the same candidate."""
    from scribblez.move_set_eval.evidence import build_evidence_inputs, collate_evidence

    max_e = 5
    # Unit 0: non-contiguous subset {0, 2}. Unit 1: the full anchor+on-policy
    # set. Unit 2: the empty subset (degrades to the plain pass).
    units = [
        (_traj_pos(_ROLES), np.array([True, False, True, False, False])),
        (_traj_pos(_ROLES), np.array([True, True, True, True, False])),
        (_traj_pos(_ROLES), np.zeros(5, bool)),
    ]
    pre_diffs = [7, -13, 4]
    batch, move_args = _cpu_evidence_batch(units, pre_diffs)
    m = int(move_args[-1].shape[0])
    plain = _fabricated_plain(m)
    fast = batch_evidence_inputs(batch, move_args, plain, max_e, torch.device("cpu"))

    items, offset = [], 0
    for (pos, mask), pre in zip(units, pre_diffs, strict=True):
        members = np.flatnonzero(mask)
        rows = offset + members
        offset += len(pos.moves)
        first = {kk: plain[kk][rows] for kk in ("wld", "score_diff", "planes")}
        items.append(
            build_evidence_inputs(pos.moves[members], pos.obs[members], pre, first, max_e=max_e)
        )
    slow = collate_evidence(items)

    import dataclasses

    for f in dataclasses.fields(fast):
        a, b = getattr(fast, f.name), getattr(slow, f.name)
        assert a.shape == b.shape, f.name
        if a.dtype == torch.bool:
            assert torch.equal(a, b), f.name
        else:
            assert torch.allclose(a.float(), b.float(), atol=1e-6), f.name
    # The empty subset carries no evidence token.
    assert not fast.mask[2].any()


def test_batch_evidence_inputs_is_membership_order_invariant():
    """Membership is a set: specifying a subset's members in a different order
    yields byte-identical EvidenceInputs (the builder canonicalizes to slot
    order), so the fusion stage's permutation invariance is never leaned on to
    hide a builder that tracked order."""
    max_e = 5
    pos = _traj_pos(_ROLES)
    forward = np.zeros(5, bool)
    forward[[0, 2, 3]] = True
    reverse = np.zeros(5, bool)
    reverse[[3, 2, 0]] = True  # same set, "built" in the other order
    plain = _fabricated_plain(len(pos.moves))
    a = batch_evidence_inputs(*_cpu_evidence_batch([(pos, forward)], [9]), plain, max_e, "cpu")
    b = batch_evidence_inputs(*_cpu_evidence_batch([(pos, reverse)], [9]), plain, max_e, "cpu")
    import dataclasses

    for f in dataclasses.fields(a):
        assert torch.equal(getattr(a, f.name), getattr(b, f.name)), f.name


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
    files = ED.complete_pairs(traj_corpus.dir)
    assert len(files) >= 2
    ED.adopt_information_condition(files)
    return ED.TrajectoryDataset(files[:-1]), ED.TrajectoryDataset(files[-1:])


@pytest.fixture(scope="module")
def mset_datasets(traj_datasets):
    """The same corpus's .mset side (the unfrozen mode's distillation rows),
    split the same way as traj_datasets."""
    from scribblez.evidence import trainer
    from scribblez.workloads import pair_store

    train, hold = traj_datasets

    def msets(ds):
        return [f.with_suffix(".mset") for f in ds.files]

    assert all(f.exists() for f in msets(train) + msets(hold))
    assert pair_store.complete_pairs(train.files[0].parent, ".mset")
    select = trainer._simmed_positions
    mset_train = MsetDataset(mset_files=msets(train), select=select)
    mset_hold = MsetDataset(mset_files=msets(hold), select=select)
    # The .mset labels more positions per game than were simmed (the fixture
    # labels 4 vs 2 simmed); the distillation side keeps the simmed ones only.
    assert 0 < mset_train.num_positions == train.num_positions
    assert MsetDataset(mset_files=msets(train)).num_positions > train.num_positions
    return mset_train, mset_hold


def test_dataset_rows_follow_the_subset(traj_datasets):
    """Every simmed candidate is a scored row; held-out is exactly ~in_evidence;
    the gain targets are the CRN-paired improvements over the subset's best; a
    subset only ever holds evidence-eligible (anchor + on-policy) records; K
    subsets per pool multiply the units."""
    train, _ = traj_datasets
    # anchor + [1..3] on-policy + the default off-policy floor (3 uniform draws).
    assert train.num_positions > 0 and train.max_trajectory <= 1 + 3 + 3
    seen_sizes = set()
    units = 0
    epochs = (b for e in range(4) for b in train.iter_batches(4, seed=1, epoch_index=e))
    for batch in epochs:
        pos_id = batch["move_pos_id"].numpy()
        in_ev = batch["in_evidence"].numpy()
        held = batch["held_out"].numpy()
        size = batch["evidence_size"].numpy()
        assert (held == ~in_ev).all()
        for p, pos in enumerate(batch["positions"]):
            rows = pos_id == p
            assert rows.sum() == len(pos.moves)
            members = in_ev[rows]
            assert int(members.sum()) == size[p]
            assert size[p] in pos.evidence_prefix_sizes()  # 0..num_evidence
            seen_sizes.add(int(size[p]))
            # Off-policy draws are labels-only: never members, always held out.
            assert not (members & ~pos.evidence_mask).any()
            _, _, value = ED.sim_targets(pos.obs)
            assert np.allclose(batch["target_gain"].numpy()[rows], ED.gain_targets(value, members))
            units += 1
    assert 0 in seen_sizes and len(seen_sizes) > 1
    # One subset per pool per epoch, over four epochs: K-multiplicity.
    assert units == 4 * train.num_positions
    triple = sum(
        1 for b in train.iter_batches(4, seed=1, subsets_per_pool=3) for _ in b["positions"]
    )
    assert triple == 3 * train.num_positions


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
    cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05, grad_clip=1.0)
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
        batch["evidence_size"].to(device)[batch["move_pos_id"].to(device)] > 0
    )
    if bool(with_ev.any()):
        assert not torch.allclose(plain["wld"][with_ev], cond["wld"][with_ev])


# --- the unfrozen mode: joint distillation + sim-outcome step ---


def _unfrozen_model(train, device):
    # num_blocks=3 so the trunk holds a real GlobalPoolingResBlock (make_block
    # emits one at every index % 3 == 2): the joint step's pooled-FC penalty
    # then collects real activations, so its wiring is exercised, not just
    # tolerated.
    torch.manual_seed(0)
    return MoveSetEvalModel(train.spatial_planes, train.scalar_size, 8, 3, 2).to(device)


def _unfrozen_params(**kw):
    return _params(unfreeze_backbone=True, **kw)


def _joint_epoch(model, opt, train, mset_train, device, lambda_sim=1.0, lr_fn=None):
    from scribblez.evidence import trainer

    cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05, grad_clip=1.0)
    distill = Distillation(
        trainer.cycle_batches(mset_train, 4, epoch=0),
        mset_train_loop.LossConfig(0.004, 10.0, 10.0, 1.0),
        lambda_sim,
    )
    return run_epoch(
        model,
        opt,
        train.iter_batches(4, seed=0),
        device,
        cfg,
        max_e=8,
        distill=distill,
        lr_fn=lr_fn,
    )


def test_unfrozen_pass_moves_the_backbone_and_keeps_prefix_0_exact(traj_datasets, mset_datasets):
    """The joint step: backbone params receive gradients and move (so the
    plain pass changes), the optimizer runs two groups at lr and
    lr * backbone_lr_mult under the schedule, the joint losses are reported,
    and prefix-0 rows stay exact between the current plain and conditioned
    passes."""
    from scribblez.evidence import trainer

    train, hold = traj_datasets
    mset_train, _ = mset_datasets
    device = torch.device("cuda")
    model = _unfrozen_model(train, device)
    assert not model.backbone_frozen
    before = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    backbone_before = [p.detach().clone() for p in model.backbone_parameters()]
    params = _unfrozen_params(lr=1e-2, backbone_lr_mult=0.1)
    opt = trainer.build_optimizer(model, params)
    assert len(opt.param_groups) == 2
    seen_lrs = []
    real_step = opt.step

    def recording_step(*a, **k):
        seen_lrs.append(tuple(g["lr"] for g in opt.param_groups))
        assert all(p.grad is not None for p in model.backbone_parameters())
        return real_step(*a, **k)

    opt.step = recording_step
    result = _joint_epoch(model, opt, train, mset_train, device, lr_fn=lambda rows: 1e-2)
    assert result.rows > 0 and np.isfinite(result.losses["total"])
    assert {"sim", "distill", "distill_wld", "distill_planes"} <= set(result.losses)
    assert result.losses["total"] == pytest.approx(
        result.losses["distill"] + result.losses["sim"], rel=1e-4
    )
    assert seen_lrs and all(lrs == (1e-2, pytest.approx(1e-3)) for lrs in seen_lrs)
    pairs = zip(backbone_before, model.backbone_parameters(), strict=True)
    assert all(not torch.equal(a, b.detach()) for a, b in pairs if b.numel() > 1)
    after = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert after["plain_wld_ce"] != pytest.approx(before["plain_wld_ce"], abs=1e-6)
    assert after["exact_p0_maxdiff"] == 0.0


def test_lambda_sim_zero_reduces_the_joint_step_to_distillation(traj_datasets, mset_datasets):
    """With lambda_sim=0 the sim loss carries no gradient: the plain outputs
    move under the distillation rows, while the proves-best head -- reached
    only through the sim loss -- stays put."""
    from scribblez.evidence import trainer

    train, hold = traj_datasets
    mset_train, _ = mset_datasets
    device = torch.device("cuda")
    model = _unfrozen_model(train, device)
    before = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    head_before = [p.detach().clone() for p in model.proves_best.parameters()]
    # No weight decay: AdamW would otherwise move a zero-gradient head too.
    opt = trainer.build_optimizer(model, _unfrozen_params(lr=1e-2, weight_decay=0.0))
    result = _joint_epoch(model, opt, train, mset_train, device, lambda_sim=0.0)
    assert result.losses["total"] == pytest.approx(result.losses["distill"], rel=1e-4)
    after = evaluate(model, hold, device, positions_per_batch=4, max_e=8)
    assert after["plain_wld_ce"] != pytest.approx(before["plain_wld_ce"], abs=1e-6)
    pairs = zip(head_before, model.proves_best.parameters(), strict=True)
    assert all(torch.equal(a, b.detach()) for a, b in pairs)


def test_mset_evaluate_loss_is_the_candidate_weighted_distillation_loss(mset_datasets):
    """evaluate(loss_cfg=...) reports the same candidate-weighted mean the
    training epoch would: recomputed here batch by batch via batch_loss."""
    from scribblez.move_set_eval import eval as mset_eval

    _, hold = mset_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(hold.spatial_planes, hold.scalar_size, 8, 1, 2).to(device).eval()
    cfg = mset_train_loop.LossConfig(0.004, 10.0, 10.0, 1.0)
    got = mset_eval.evaluate(model, hold, device, positions_per_batch=3, loss_cfg=cfg)
    total = wld = 0.0
    n = 0
    with torch.no_grad():
        for batch in hold.iter_batches(
            3, seed=0, max_candidates=mset_eval.MAX_CANDIDATES_PER_BATCH
        ):
            losses = mset_train_loop.batch_loss(model, batch, device, cfg)
            m = batch["target_wld"].shape[0]
            total += losses["total"].item() * m
            wld += losses["wld"].item() * m
            n += m
    assert n > 3  # more than one batch, so the weighting is exercised
    assert got["loss"] == pytest.approx(total / n, rel=1e-5)
    assert got["loss_wld"] == pytest.approx(wld / n, rel=1e-5)


def test_distill_datasets_refuse_an_empty_selection(traj_datasets, monkeypatch):
    """A .sobs/.mset pairing whose selection matches nothing fails at load
    (a starved joint step would otherwise hang on its first batch)."""
    from scribblez.evidence import trainer

    train, _ = traj_datasets
    monkeypatch.setattr(trainer, "_simmed_positions", lambda path: set())
    with pytest.raises(ValueError, match="no trajectory position"):
        trainer._simmed_mset([f.with_suffix(".mset") for f in train.files])


def test_frozen_optimizer_has_one_group_and_no_backbone(traj_datasets):
    from scribblez.evidence import trainer

    train, _ = traj_datasets
    model = _tiny_model()
    model.freeze_backbone()
    opt = trainer.build_optimizer(model, _params(lr=1e-3))
    assert len(opt.param_groups) == 1
    assert {p.data_ptr() for p in opt.param_groups[0]["params"]} == {
        p.data_ptr() for p in model.evidence_parameters()
    }


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
    # The .mset side is the same split of stems, listing only the stems that
    # carry a .mset.
    for f in train_files[:2] + hold_files[:1]:
        f.with_suffix(".mset").touch()
    mset_train, mset_hold = trainer.split_pairs(store, 4, ".mset")
    assert mset_train == [f.with_suffix(".mset") for f in train_files[:2]]
    assert mset_hold == [hold_files[0].with_suffix(".mset")]
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

    def read_json(self, rel):
        return None


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
    # subsets_per_pool > 1 so arbitrary (non-prefix) subsets are exercised: the
    # per-position builder is fed the subset's members, and the batched builder
    # must pack them the same compact way.
    for batch in train.iter_batches(4, seed=2, subsets_per_pool=3):
        move_args = tuple(batch[k].to(device) for k in _MOVE_KEYS)
        spatial, scalar = (batch[k].to(device) for k in _INPUT_KEYS)
        pos_id = move_args[-1]
        in_ev = batch["in_evidence"].numpy()
        with torch.no_grad():
            board, g = model.encode_board(spatial, scalar)
            plain = model.score_moves(board, g, model.encode_moves(board, *move_args), pos_id)
            fast = batch_evidence_inputs(batch, move_args, plain, max_e, device)
            items = []
            offset = 0
            for p, pos in enumerate(batch["positions"]):
                members = np.flatnonzero(in_ev[offset : offset + len(pos.moves)])
                rows = torch.from_numpy(offset + members).to(device)
                offset += len(pos.moves)
                first = {kk: plain[kk][rows] for kk in ("wld", "score_diff", "planes")}
                items.append(
                    build_evidence_inputs(
                        pos.moves[members],
                        pos.obs[members],
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
    from scribblez.move_set_eval.moves import move_encoding_version

    train, _ = traj_datasets
    store_src = traj_corpus.dir
    tag = "zz-evidence-run"
    ckpt = tmp_path / "student.pt"
    _student_checkpoint(ckpt, train, open_leaves=False)
    # holdout_every=0: the corpus is one or two files, so a hashed split could
    # leave the training side empty; on-train metrics suffice for the runner.
    base = dict(
        proposer_model="/x",
        teacher_model="/x",
        student_checkpoint=str(ckpt),
        warmup_pairs=1,
        holdout_every=0,
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
        for ext in (".sobs", ".slog", ".mset"):
            shutil.copy(f.with_suffix(ext), store / f.with_suffix(ext).name)
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
    # Frozen: the plain model is the student, so no per-pass ONNX.
    assert not scratch.paths.onnx_dir.exists()
    import sqlite3

    conn = sqlite3.connect(scratch.paths.dashboard_db)
    epochs = {r[0] for r in conn.execute("select epoch from metrics")}
    names = {r[0] for r in conn.execute("select name from metrics")}
    assert epochs == {0, 1}
    assert {"loss", "cond_wld_ce", "plain_wld_ce", "gain_hit_acc", "exact_p0_maxdiff"} <= names
    assert not names & {"student_wld_ce", "distill_recall1", "loss_distill"}
    # Resume: the budget is spent, so the second run stops without a pass.
    assert trainer.run(scratch.ctx) == 0
    assert not (scratch.paths.checkpoints_dir / "model_epoch_0002.pt").exists()

    # The unfrozen mode over the same store, as its own tag: per-pass ONNX,
    # the joint losses, the flat student reference, the distillation health
    # series, prefix-0 exactness, and the mode recorded in the checkpoint.
    tag_u = "zz-evidence-run-unfrozen"
    scratch_u = _ctx(tmp_path, tag_u, _params(train_epochs=2, unfreeze_backbone=True, **base))
    store_u = scratch_u.paths.data_dir / "slogs"
    shutil.copytree(store, store_u)
    for f in store_u.iterdir():
        os.utime(f, (stale, stale))
    assert trainer.run(scratch_u.ctx) == 0
    import onnx

    for epoch in (0, 1):
        assert (scratch_u.paths.checkpoints_dir / f"model_epoch_{epoch:04d}.pt").exists()
        # The export is stamped with the student's arm and version, not the
        # session's or a default's.
        meta = {
            e.key: e.value for e in onnx.load(str(scratch_u.paths.onnx_path(epoch))).metadata_props
        }
        assert meta["graph"] == "move_set_eval", meta
        assert meta["opp_leave_input"] == "false", meta
        assert meta["move_encoding_version"] == str(move_encoding_version()), meta
    conn = sqlite3.connect(scratch_u.paths.dashboard_db)
    rows = {}
    for epoch, name, value in conn.execute("select epoch, name, value from metrics"):
        rows.setdefault(name, {})[epoch] = value
    assert {
        "loss_sim",
        "loss_distill",
        "loss_distill_wld",
        "student_wld_ce",
        "distill_recall1",
        "distill_spearman",
        "distill_loss",
        "exact_p0_maxdiff",
    } <= set(rows)
    assert rows["student_wld_ce"][0] == rows["student_wld_ce"][1]
    assert rows["plain_wld_ce"][0] != rows["plain_wld_ce"][1]
    assert all(v == 0.0 for v in rows["exact_p0_maxdiff"].values())
    ckpt = torch.load(scratch_u.paths.checkpoints_dir / "model_epoch_0001.pt", weights_only=False)
    assert ckpt["config"]["unfreeze_backbone"] is True
    assert ckpt["config"]["backbone_lr_mult"] == 0.1


# --- divergence guards ---


@pytest.mark.parametrize("frozen", [True, False])
def test_non_finite_batches_are_skipped_without_a_step(
    traj_datasets, mset_datasets, monkeypatch, frozen
):
    """A batch whose loss is non-finite takes no optimizer step and is counted;
    the pass otherwise proceeds -- in both modes (the unfrozen step is joint,
    and a poisoned sim loss poisons its total)."""
    from scribblez.evidence import train_loop, trainer

    train, _ = traj_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(train.spatial_planes, train.scalar_size, 8, 1, 2).to(device)
    if frozen:
        model.freeze_backbone()
    real = train_loop.compute_loss
    calls = {"n": 0}

    def poison_every_other(outputs, targets, cfg):
        calls["n"] += 1
        losses = real(outputs, targets, cfg)
        if calls["n"] % 2 == 0:
            losses = {k: v * float("nan") for k, v in losses.items()}
        return losses

    monkeypatch.setattr(train_loop, "compute_loss", poison_every_other)
    before = [p.detach().clone() for p in model.parameters()]
    opt = trainer.build_optimizer(model, _params(unfreeze_backbone=not frozen, lr=1e-2))
    if frozen:
        cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05, grad_clip=1.0)
        result = run_epoch(model, opt, train.iter_batches(4, seed=0), device, cfg, max_e=8)
    else:
        result = _joint_epoch(model, opt, train, mset_datasets[0], device)
    assert result.skipped >= 1 and result.n_batches >= 1
    assert result.skipped + result.n_batches == calls["n"]
    assert np.isfinite(result.losses["total"])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    pairs = zip(before, model.parameters(), strict=True)
    assert any(not torch.equal(a, b.detach()) for a, b in pairs)  # the good batches stepped


def test_check_finite_stops_a_diverged_pass():
    from scribblez.evidence import trainer
    from scribblez.evidence.train_loop import EpochResult

    model = _tiny_model()
    ok = EpochResult({"total": 0.5}, 10, 100, 100, skipped=0)
    trainer.check_finite(model, ok)  # healthy: no-op
    with pytest.raises(RuntimeError, match="skipped|non-finite loss"):
        trainer.check_finite(model, EpochResult({"total": 0.5}, 10, 100, 100, skipped=99))
    with torch.no_grad():
        model.proves_best[0].weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite parameters"):
        trainer.check_finite(model, ok)


@pytest.mark.parametrize("frozen", [True, False])
def test_gradients_are_clipped_to_the_configured_norm(traj_datasets, mset_datasets, frozen):
    """With clipping on, the total gradient norm handed to the optimizer never
    exceeds grad_clip (checked by wrapping optimizer.step) -- over the
    evidence params when frozen, over every param when unfrozen."""
    from scribblez.evidence import trainer

    train, _ = traj_datasets
    device = torch.device("cuda")
    model = MoveSetEvalModel(train.spatial_planes, train.scalar_size, 8, 1, 2).to(device)
    if frozen:
        model.freeze_backbone()
    opt = trainer.build_optimizer(model, _params(unfreeze_backbone=not frozen, lr=1e-2))
    params = [p for g in opt.param_groups for p in g["params"]]
    seen = []
    real_step = opt.step

    def recording_step(*a, **k):
        seen.append(
            float(torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])))
        )
        return real_step(*a, **k)

    opt.step = recording_step
    cfg = LossConfig(0.004, 1.0, 10.0, 10.0, 0.05, grad_clip=0.01)
    distill = None
    if not frozen:
        distill = Distillation(
            trainer.cycle_batches(mset_datasets[0], 4, epoch=0),
            mset_train_loop.LossConfig(0.004, 10.0, 10.0, 1.0),
            1.0,
        )
    run_epoch(model, opt, train.iter_batches(4, seed=0), device, cfg, max_e=8, distill=distill)
    assert seen and max(seen) <= 0.01 * (1 + 1e-4)
