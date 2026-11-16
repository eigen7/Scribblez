"""The deep_bot workload (scribblez.workloads.deep_bot): a new tag's seeds,
the generator's cycle on the newest release, and the tag's end condition."""

from dataclasses import replace

import pytest
from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.deep_bot import KINDS, LEAF, LEAVE, release, seeds
from scribblez.generational import lifecycle
from scribblez.params import ParamsError
from scribblez.paths import TagPaths
from scribblez.workloads import deep_bot, selfplay_gen
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


def _ctx(tmp_path, max_cycles: int) -> WorkerContext:
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
    )


def test_each_cycle_plays_and_labels_on_the_newest_release(tmp_path, monkeypatch):
    paths = SPEC.paths("t", tmp_path)
    _publish(paths, 0)
    played, labeled = [], []

    def run_games(out_dir, player_spec, **kwargs):
        played.append(player_spec)
        (out_dir / f"{out_dir.name}.slog").write_bytes(b"slog")
        if len(played) == 1:
            _publish(paths, 1)  # the trainer publishes while the first chunk plays
        return 0

    def write_pools(pool_params, leaf, leave, chunk_dir, threads):
        labeled.append((pool_params.positions_per_game, leaf, leave))
        for slog in chunk_dir.glob("*.slog"):
            for suffix in deep_bot.CHUNK_SIDECARS:
                slog.with_suffix(suffix).write_bytes(b"side")

    monkeypatch.setattr(selfplay_gen, "run_games", run_games)
    monkeypatch.setattr(deep_bot, "write_pools", write_pools)

    assert deep_bot.run_generate(_ctx(tmp_path, max_cycles=2)) == 0

    # The first chunk was planned before release 1 existed; the second after.
    for cycle, number in enumerate((0, 1)):
        leaf = release.export_path(paths, number, LEAF)
        leave = release.export_path(paths, number, LEAVE)
        assert f"--leaf-model {leaf} --leave-inference {leave}" in played[cycle]
        assert "--type=deepbot-endgame --reply hasty" in played[cycle]
        assert labeled[cycle] == (PARAMS.pool_positions_per_game, leaf, leave)
    staged = sorted(p.suffix for p in (paths.data_dir / "staging").iterdir())
    assert staged == sorted([*deep_bot.CHUNK_SIDECARS, ".slog"] * 2)


def test_a_generator_without_a_release_fails(tmp_path):
    with pytest.raises(RuntimeError, match="no release"):
        deep_bot.release_cycle(SPEC.paths("t", tmp_path), PARAMS)


def test_the_tag_is_complete_at_max_generations(tmp_path):
    paths = SPEC.paths("t", tmp_path)
    bounded = replace(PARAMS, max_generations=3)
    assert not SPEC.is_complete(paths, bounded)  # no trainer state yet

    lifecycle.write_train_state(paths, {"generation_index": 3, "rows_trained": 6000})
    _publish(paths, 3)

    assert SPEC.is_complete(paths, bounded)
    assert not SPEC.is_complete(paths, PARAMS)  # unbounded
    assert deep_bot.progress(SPEC, paths, PARAMS) == [("release", 3), ("games", 6000)]
