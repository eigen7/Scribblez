"""Which worker image a role runs on (RoleSpec.runtime -> RegistryConfig.
image_for), and how a train role gets its eval datasets onto a bundle-run
worker (cloud/bundles.py deps/ + cloud/worker_deps.py)."""

import shutil
import tarfile

import pytest
from cloud import bundles, worker_deps
from cloud.credentials import RegistryConfig
from cloud.runtime_abi import RUNTIME_ENGINE, RUNTIME_TORCH
from cloud.worker_entrypoint import WORKER_ENV_VARS
from scribblez import workloads
from scribblez.workloads.base import RoleSpec, WorkloadSpec
from scribblez.workloads.kill_test import KillTestParams


@pytest.mark.parametrize(
    "image, torch_image",
    [
        ("docker.io/u/scribblez", "docker.io/u/scribblez:latest-torch"),
        ("docker.io/u/scribblez:v3", "docker.io/u/scribblez:v3-torch"),
        ("registry.local:5000/scribblez", "registry.local:5000/scribblez:latest-torch"),
        ("scribblez", "scribblez:latest-torch"),
    ],
)
def test_the_torch_image_is_the_engine_image_under_a_suffixed_tag(image, torch_image):
    registry = RegistryConfig(worker_image=image)
    assert registry.image_for(RUNTIME_ENGINE) == image
    assert registry.image_for(RUNTIME_TORCH) == torch_image


def test_train_roles_take_the_torch_runtime_and_the_rest_the_engine():
    for spec in workloads.WORKLOADS.values():
        for role in spec.roles:
            assert role.runtime == (RUNTIME_TORCH if role.name == "train" else RUNTIME_ENGINE), (
                spec.name,
                role.name,
            )


def test_a_role_naming_no_such_runtime_is_refused():
    with pytest.raises(AssertionError, match="no such runtime"):
        WorkloadSpec(
            name="x",
            title="x",
            params_cls=KillTestParams,
            roles=(RoleSpec(name="r", title="r", runner="a:b", runtime="cuda"),),
        )


def test_the_trainer_device_is_a_worker_variable_not_a_param():
    assert "SCZ_DEVICE" in WORKER_ENV_VARS


# --- eval datasets ---------------------------------------------------------


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    """Two small eval datasets under a fake repo root, as EVAL_POSITIONS_DIRS."""
    root = tmp_path / "repo"
    dirs = (root / "positions" / "NWL23" / "small", root / "positions" / "NWL23" / "large")
    for d in dirs:
        d.mkdir(parents=True)
        (d / "pos-01.gcg").write_text(f"#{d.name}\n")
    (dirs[1] / "part-000.gcgs").write_text("bundle\n")
    monkeypatch.setattr(bundles, "REPO_ROOT", root)
    monkeypatch.setattr(bundles, "EVAL_POSITIONS_DIRS", dirs)
    monkeypatch.setattr(worker_deps, "EVAL_POSITIONS_DIRS", dirs)
    return root, dirs


def test_the_datasets_digest_is_content_addressed(datasets):
    root, dirs = datasets
    first = bundles.eval_positions_digest()
    assert first == bundles.eval_positions_digest()
    (dirs[0] / "pos-01.gcg").write_text("changed\n")
    assert bundles.eval_positions_digest() != first
    assert bundles.eval_positions_path(root, "abc") == root / "deps" / "positions-abc.tar.gz"


def test_the_datasets_tarball_unpacks_under_a_repo_root(datasets, tmp_path):
    root, dirs = datasets
    tar_path = bundles.create_eval_positions_tarball(tmp_path)
    with tarfile.open(tar_path) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(str(p.relative_to(root)) for _, p in bundles.eval_positions_files())


def test_the_store_writes_a_version_once(datasets, tmp_path):
    digest = bundles.eval_positions_digest()
    bundles.write_eval_positions(tmp_path, digest)
    path = bundles.eval_positions_path(tmp_path, digest)
    stamp = path.stat().st_mtime_ns
    bundles.write_eval_positions(tmp_path, digest)  # content-addressed: nothing to redo
    assert path.stat().st_mtime_ns == stamp
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no work dir left


def test_a_worker_needs_its_datasets(datasets, monkeypatch):
    """A local slot or CLI uses the checkout's copy, and a container the copy
    its bootstrap unpacked; either way it must be there."""
    root, dirs = datasets
    worker_deps.fetch_eval_positions()  # present: fine
    shutil.rmtree(dirs[1])
    with pytest.raises(AssertionError, match="eval datasets missing"):
        worker_deps.fetch_eval_positions()


def test_the_trainer_refuses_to_start_without_its_eval_datasets(tmp_path, monkeypatch):
    """A run without its eval curves is not the run anyone asked for; on a
    rented machine it would train for hours before anyone noticed. So a
    missing dataset is a startup error, not a log line."""
    pytest.importorskip("torch")
    from scribblez.position_eval import analysis, trainer

    monkeypatch.setattr(analysis, "DEFAULT_DATASET", tmp_path / "absent")
    monkeypatch.setattr(analysis, "LARGE_DATASET", tmp_path / "absent-large")
    with pytest.raises(Exception, match="absent"):
        trainer.load_position_eval_quality(87, face_up_leaves=True)
