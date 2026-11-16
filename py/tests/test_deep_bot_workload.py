"""The deep_bot workload (scribblez.workloads.deep_bot): a new tag's seeds,
the generator's cycle on the newest release, what the controller carries to
slots on other machines, and the tag's end condition."""

from dataclasses import replace

import pytest
from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.deep_bot import KINDS, LEAF, LEAVE, release, seeds
from scribblez.generational import lifecycle
from scribblez.leave_inference import forgone
from scribblez.params import ParamsError
from scribblez.paths import TagPaths
from scribblez.policy_set import POOL_SUFFIX, SURVIVORS_SUFFIX
from scribblez.workloads import base, deep_bot, selfplay_gen
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.deep_bot import DeepBotParams

SPEC = workloads.get("deep_bot")
PARAMS = DeepBotParams(seed_leave_tag="i", seed_leaf_tag="l", seed_policy_tag="p")


def _seed_tag(mount, kind: str) -> TagPaths:
    """`kind`'s seed tag under `mount`, with a stand-in rolling checkpoint."""
    paths = TagPaths(getattr(PARAMS, f"seed_{kind}_tag"), seeds.WORKLOADS[kind], mount)
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    paths.rolling_checkpoint.write_bytes(kind.encode())
    return paths


def test_creating_a_tag_copies_its_seeds(tmp_path):
    for kind in KINDS:
        _seed_tag(tmp_path, kind)
    paths = SPEC.paths("t", tmp_path)

    assert deep_bot.finalize(SPEC, paths, PARAMS) == PARAMS

    for kind in KINDS:
        assert seeds.seed_path(paths, kind).read_bytes() == kind.encode()


def test_a_missing_seed_refuses_the_tag_before_anything_is_copied(tmp_path):
    _seed_tag(tmp_path, LEAVE)
    paths = SPEC.paths("t", tmp_path)

    with pytest.raises(ParamsError, match="deep_bot_leaf_eval tag 'l' has no checkpoint"):
        deep_bot.finalize(SPEC, paths, PARAMS)
    with pytest.raises(ParamsError, match="seed_policy_tag is required"):
        seeds.source(replace(PARAMS, seed_policy_tag=""), "policy", tmp_path)

    assert not seeds.seed_path(paths, LEAVE).exists()


def _publish(paths: TagPaths, number: int):
    release.release_dir(paths, number).mkdir(parents=True)
    release.write_manifest(paths, number, "t", {})


def _ctx(tmp_path, max_cycles: int = 1, kind: str = "local") -> WorkerContext:
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role("generate"),
        tag="t",
        params=PARAMS,
        worker_id="local-0",
        threads=2,
        max_cycles=max_cycles,
        sink=LocalSink(SPEC.data_dir("t", tmp_path)),
        mount_root=tmp_path,
        kind=kind,
    )


def test_each_cycle_plays_the_newest_release_and_keeps_its_own_pools(tmp_path, monkeypatch):
    paths = SPEC.paths("t", tmp_path)
    _publish(paths, 0)
    played = []

    def run_games(out_dir, player_spec, pool_positions_per_game, **kwargs):
        """play_game's part of a chunk: the games, and the pool sidecars of
        the turns the agent searched."""
        played.append((player_spec, pool_positions_per_game))
        slog = out_dir / f"{out_dir.name}.slog"
        for suffix in (".slog", POOL_SUFFIX, SURVIVORS_SUFFIX):
            slog.with_suffix(suffix).write_bytes(b"x")
        if len(played) == 1:
            _publish(paths, 1)  # the trainer publishes while the first chunk plays
        return 0

    def write_forgone(chunk_dir, threads):
        for slog in chunk_dir.glob("*.slog"):
            slog.with_suffix(forgone.SUFFIX).write_bytes(b"x")

    monkeypatch.setattr(selfplay_gen, "run_games", run_games)
    monkeypatch.setattr(forgone, "write_sidecars", write_forgone)

    assert deep_bot.run_generate(_ctx(tmp_path, max_cycles=2)) == 0

    # The first chunk was planned before release 1 existed; the second after.
    for cycle, number in enumerate((0, 1)):
        leaf = release.export_path(paths, number, LEAF)
        leave = release.export_path(paths, number, LEAVE)
        spec, kept = played[cycle]
        assert f"--leaf-model {leaf} --leave-inference {leave}" in spec
        assert "--type=deepbot-endgame --reply hasty" in spec
        assert kept == PARAMS.pool_positions_per_game
    staged = sorted(p.suffix for p in (paths.data_dir / "staging").iterdir())
    assert staged == sorted([*deep_bot.CHUNK_SIDECARS, ".slog"] * 2)


def test_a_generator_without_a_release_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="no release"):
        deep_bot.release_cycle(_ctx(tmp_path))


def test_a_remote_generator_waits_for_the_controllers_first_release(tmp_path, monkeypatch):
    paths = SPEC.paths("t", tmp_path)
    # The push lands while the generator sleeps.
    monkeypatch.setattr(base.time, "sleep", lambda seconds: _publish(paths, 4))
    assert deep_bot.newest_release(_ctx(tmp_path, kind="ssh")) == 4


def test_the_controller_carries_seeds_to_a_trainer_and_the_newest_release_to_generators(tmp_path):
    paths = SPEC.paths("t", tmp_path)
    tag_files = {r.name: workloads.resolve(r.tag_files) for r in SPEC.roles if r.tag_files}
    assert tag_files["train"](paths, PARAMS) == [f"seeds/{kind}.pt" for kind in KINDS]
    assert tag_files["generate"](paths, PARAMS) == []  # before release 0
    _publish(paths, 0)
    _publish(paths, 1)
    # The manifest goes last: a release exists once it does.
    assert tag_files["generate"](paths, PARAMS) == [
        *(f"releases/release_000001/{kind}.onnx" for kind in KINDS),
        "releases/release_000001/manifest.json",
    ]


def test_the_controller_keeps_generations_for_the_longest_window():
    assert replace(PARAMS, leave_window=2, leaf_window=5, policy_window=3).window == 5


def test_the_tag_is_complete_at_max_generations(tmp_path):
    paths = SPEC.paths("t", tmp_path)
    bounded = replace(PARAMS, max_generations=3)
    assert not SPEC.is_complete(paths, bounded)  # no trainer state yet

    lifecycle.write_train_state(paths, {"generation_index": 3, "rows_trained": 6000})
    _publish(paths, 3)

    assert SPEC.is_complete(paths, bounded)
    assert not SPEC.is_complete(paths, PARAMS)  # unbounded
    assert deep_bot.progress(SPEC, paths, PARAMS) == [("release", 3), ("games trained on", 6000)]
