"""Placement rules (dashboard/placement.py) and position_eval's layout and GPU
table: which machine may take which queued tag, and how queued tags are matched
to free machines."""

from dataclasses import replace

from scribblez.dashboard import placement
from scribblez.dashboard.pool import Hardware, PoolMachine
from scribblez.dashboard.queue import BUNDLE_BUILDING, BUNDLE_READY, QueueEntry
from scribblez.workloads import position_eval as pe
from scribblez.workloads.position_eval import SPEC, PositionEvalParams

# The transformer profile's trainer, as the tuning campaign runs it.
CAMPAIGN = PositionEvalParams(
    trunk="transformer",
    grad_clip=1.0,
    activation_checkpointing=False,
    max_rows=24_000_000,
    match_every_generations=0,
)


def _machine(name, kind="ssh", gpu_gb=22.5, reserve=0.0, vcpus=8) -> PoolMachine:
    return PoolMachine(
        name=name, kind=kind, hardware=Hardware(vcpus, 1 if gpu_gb else 0, gpu_gb),
        gpu_reserve_gb=reserve,
    )  # fmt: skip


def _entry(tag="t", **kw) -> QueueEntry:
    return QueueEntry("position_eval", tag, 0.0, bundle=kw.pop("bundle", BUNDLE_READY), **kw)


def test_the_campaign_trainer_needs_its_measured_peak_plus_headroom():
    assert pe.gpu_need(CAMPAIGN, "train") == 13.53 + pe.TRAINER_GPU_HEADROOM_GB
    checkpointed = replace(CAMPAIGN, activation_checkpointing=True)
    assert pe.gpu_need(checkpointed, "train") == 8.46 + pe.TRAINER_GPU_HEADROOM_GB
    assert pe.gpu_need(CAMPAIGN, "generate") == 0.0
    assert pe.gpu_need(CAMPAIGN, "match_eval") == pe.MATCH_EVAL_GPU_GB
    # Unmeasured: a batch size other than the one measured.
    assert pe.gpu_need(replace(CAMPAIGN, batch_size=512), "train") is None


def test_the_layout_gives_the_generator_the_machine():
    plan = pe.layout(CAMPAIGN, 28, None)
    assert [(p.role, p.threads) for p in plan] == [("train", None), ("generate", 28)]
    assert [p.threads for p in pe.layout(CAMPAIGN, 28, 20) if p.role == "generate"] == [20]
    with_matches = replace(CAMPAIGN, match_every_generations=5)
    assert [p.role for p in pe.layout(with_matches, 8, None)] == ["train", "generate", "match_eval"]


def test_gpu_memory_decides_eligibility():
    """The asus-laptop OOM: a 4 GiB GPU cannot take the campaign trainer, an
    L4 can, and the 16 GiB laptop can once its reserve is counted."""
    params = CAMPAIGN
    assert "needs 14.0 GiB" in placement.refusal(
        SPEC, params, _entry(), _machine("asus", gpu_gb=4.0)
    )
    assert placement.refusal(SPEC, params, _entry(), _machine("l4")) is None
    local = _machine("localhost", "local", gpu_gb=16.0, reserve=1.7)
    assert placement.refusal(SPEC, params, _entry(), local) is None
    assert "has 12.3" in placement.refusal(
        SPEC, params, _entry(), replace(local, gpu_reserve_gb=3.7)
    )


def test_an_unmeasured_config_waits_for_an_override():
    params = replace(CAMPAIGN, batch_size=512)
    assert "no GPU memory figure" in placement.refusal(SPEC, params, _entry(), _machine("l4"))
    assert placement.refusal(SPEC, params, _entry(memory_override_gb=18.0), _machine("l4")) is None


def test_match_eval_shares_the_gpu():
    """Trainer and match eval sum on one GPU: 16.6 GiB fits an L4 but not the
    16 GiB laptop, which fits the trainer alone."""
    params = replace(CAMPAIGN, match_every_generations=5)
    assert placement.refusal(SPEC, params, _entry(), _machine("l4")) is None
    laptop = _machine("localhost", "local", gpu_gb=16.0, reserve=1.7)
    assert "needs 16.6 GiB" in placement.refusal(SPEC, params, _entry(), laptop)


def test_named_machines_and_bundles():
    assert "not among" in placement.refusal(
        SPEC, CAMPAIGN, _entry(machines=["other"]), _machine("l4")
    )
    building = _entry(bundle=BUNDLE_BUILDING)
    assert "bundle is building" in placement.refusal(SPEC, CAMPAIGN, building, _machine("l4"))
    assert placement.refusal(SPEC, CAMPAIGN, building, _machine("l4"), need_bundle=False) is None
    # Local slots run the checkout, so localhost needs no bundle.
    local = _machine("localhost", "local", gpu_gb=16.0)
    assert placement.refusal(SPEC, CAMPAIGN, building, local) is None


def test_matching_moves_an_earlier_tag_so_a_later_one_starts():
    """A fits both machines, B only the rental. Greedy in machine order would
    give A the rental and leave B waiting beside a free laptop."""
    a, b = _entry("a"), _entry("b")
    rental, laptop = _machine("rental"), _machine("laptop")
    fits = {("a", "rental"), ("a", "laptop"), ("b", "rental")}
    matches = placement.match([a, b], [rental, laptop], lambda e, m: (e.tag, m.name) in fits)
    assert matches == {a.key: "laptop", b.key: "rental"}


def test_matching_never_drops_an_earlier_tag_for_a_later_one():
    a, b = _entry("a"), _entry("b")
    only = _machine("only")
    assert placement.match([a, b], [only], lambda e, m: True) == {a.key: "only"}
    assert placement.match([b, a], [only], lambda e, m: True) == {b.key: "only"}


def test_end_condition():
    assert placement.has_end_condition(SPEC, CAMPAIGN)
    assert not placement.has_end_condition(SPEC, replace(CAMPAIGN, max_rows=-1))
    assert not placement.has_end_condition(SPEC, replace(CAMPAIGN, max_rows=0))
