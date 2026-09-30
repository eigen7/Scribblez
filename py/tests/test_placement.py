"""Placement rules (dashboard/placement.py) and position_eval's layout and GPU
table: which machine may take which queued tag, and how queued tags are matched
to free machines."""

from dataclasses import replace

from scribblez.dashboard import placement, tasks
from scribblez.dashboard.pool import Hardware, PoolMachine
from scribblez.dashboard.queue import BUNDLE_BUILDING, BUNDLE_READY, QueueEntry
from scribblez.paths import POSITION_EVAL, TagPaths
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


def _machine(name, kind="ssh", gpu_gb=22.5, vcpus=8) -> PoolMachine:
    return PoolMachine(name=name, kind=kind, hardware=Hardware(vcpus, 1 if gpu_gb else 0, gpu_gb))


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
    L4 can, and so can the 16 GiB laptop, but not a 12.3 GiB one."""
    params = CAMPAIGN
    assert "needs 14.0 GiB" in placement.refusal(
        SPEC, params, None, _entry(), _machine("asus", gpu_gb=4.0)
    )
    assert placement.refusal(SPEC, params, None, _entry(), _machine("l4")) is None
    local = _machine("localhost", "local", gpu_gb=16.0)
    assert placement.refusal(SPEC, params, None, _entry(), local) is None
    small = _machine("localhost", "local", gpu_gb=12.3)
    assert "has 12.3" in placement.refusal(SPEC, params, None, _entry(), small)


def test_an_unmeasured_config_waits_for_an_override():
    params = replace(CAMPAIGN, batch_size=512)
    assert "no GPU memory figure" in placement.refusal(SPEC, params, None, _entry(), _machine("l4"))
    override = _entry(memory_override_gb=18.0)
    assert placement.refusal(SPEC, params, None, override, _machine("l4")) is None


def test_match_eval_shares_the_gpu():
    """Trainer and match eval sum on one GPU: 16.6 GiB fits an L4 but not the
    16 GiB laptop, which fits the trainer alone."""
    params = replace(CAMPAIGN, match_every_generations=5)
    assert placement.refusal(SPEC, params, None, _entry(), _machine("l4")) is None
    laptop = _machine("localhost", "local", gpu_gb=16.0)
    assert "needs 16.6 GiB" in placement.refusal(SPEC, params, None, _entry(), laptop)


def test_named_machines_and_bundles():
    assert "not among" in placement.refusal(
        SPEC, CAMPAIGN, None, _entry(machines=["other"]), _machine("l4")
    )
    building = _entry(bundle=BUNDLE_BUILDING)
    assert "bundle is building" in placement.refusal(SPEC, CAMPAIGN, None, building, _machine("l4"))
    l4 = _machine("l4")
    assert placement.refusal(SPEC, CAMPAIGN, None, building, l4, need_bundle=False) is None
    # Local slots run the checkout, so localhost needs no bundle.
    local = _machine("localhost", "local", gpu_gb=16.0)
    assert placement.refusal(SPEC, CAMPAIGN, None, building, local) is None


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


def test_a_tag_goes_only_where_its_training_state_is():
    """State only on localhost (a local trainer's): an ssh machine would start
    the trainer over and then overwrite the local checkpoint. State in the
    bucket, or none yet: anywhere."""
    local = _machine("localhost", "local", gpu_gb=16.0)
    rental = _machine("l4")
    home = placement.HOME_LOCAL
    assert "training state is only on localhost" in placement.refusal(
        SPEC, CAMPAIGN, home, _entry(), rental
    )
    assert placement.refusal(SPEC, CAMPAIGN, home, _entry(), local) is None
    for home in (placement.HOME_BUCKET, None):
        assert placement.refusal(SPEC, CAMPAIGN, home, _entry(), rental) is None
        assert placement.refusal(SPEC, CAMPAIGN, home, _entry(), local) is None


def test_a_data_home_tag_stays_on_localhost_from_the_start(tmp_path):
    """Its trainer must be a local slot, before it has trained a row as after."""
    task = tasks.TaskRecord(workload="position_eval", tag="t", params={}, created_at=0.0)
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    assert placement.state_home(paths, task) is None
    task.data_plane = "home"
    assert placement.state_home(paths, task) == placement.HOME_LOCAL
