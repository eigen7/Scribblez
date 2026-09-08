"""Which worker image a role runs on (RoleSpec.runtime -> RegistryConfig.
image_for), and how a train role gets its eval datasets onto a bundle-run
worker (cloud/bundles.py deps/ + cloud/worker_deps.py)."""

import os
import tarfile
from pathlib import Path

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
    monkeypatch.setattr(worker_deps, "REPO_ROOT", root)
    monkeypatch.setattr(worker_deps, "EVAL_POSITIONS_DIRS", dirs)
    return root, dirs


def test_the_datasets_digest_is_content_addressed(datasets):
    root, dirs = datasets
    first = bundles.eval_positions_digest()
    assert first == bundles.eval_positions_digest()
    (dirs[0] / "pos-01.gcg").write_text("changed\n")
    assert bundles.eval_positions_digest() != first
    assert bundles.eval_positions_object("abc") == "deps/positions-abc.tar.gz"


def test_the_datasets_tarball_unpacks_under_a_repo_root(datasets, tmp_path):
    root, dirs = datasets
    tar_path = bundles.create_eval_positions_tarball(tmp_path)
    with tarfile.open(tar_path) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(str(p.relative_to(root)) for _, p in bundles.eval_positions_files())


class _Bucket:
    """rclone over a dict: what push and fetch ask of the bucket."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def rclone(self, r2, *args, capture=False, input_text=None):
        from types import SimpleNamespace

        op, rest = args[0], args[1:]
        if op == "lsf":
            key = rest[0].split(":", 1)[1].split("/", 1)[1]
            out = key.rsplit("/", 1)[-1] + "\n" if key in self.objects else ""
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        if op == "copyto":
            src, dst = rest
            if src.startswith("r2:"):
                key = src.split(":", 1)[1].split("/", 1)[1]
                if key not in self.objects:
                    return SimpleNamespace(returncode=1, stdout="", stderr="not found")
                Path(dst).write_bytes(self.objects[key])
            else:
                key = dst.split(":", 1)[1].split("/", 1)[1]
                self.objects[key] = Path(src).read_bytes()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected rclone {args}")


@pytest.fixture
def bucket(monkeypatch):
    b = _Bucket()
    monkeypatch.setattr(bundles, "rclone", b.rclone)
    monkeypatch.setattr(worker_deps, "rclone", b.rclone)
    return b


def test_push_uploads_a_version_once(datasets, bucket):
    from cloud.credentials import R2Credentials

    r2 = R2Credentials(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
    digest = bundles.eval_positions_digest()
    bundles.push_eval_positions(r2, digest)
    key = bundles.eval_positions_object(digest)
    assert key in bucket.objects
    stored = bucket.objects[key]
    bundles.push_eval_positions(r2, digest)  # content-addressed: nothing to redo
    assert bucket.objects[key] is stored


def test_a_bundle_run_worker_takes_the_manifest_version(datasets, bucket, monkeypatch, tmp_path):
    """The worker's copy is replaced when it is at another version than the
    bundle names, and left alone when it matches; a bundle naming none is
    refused."""
    from cloud.credentials import R2Credentials

    root, dirs = datasets
    r2 = R2Credentials(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
    monkeypatch.setattr(worker_deps, "r2_from_env", lambda: r2)
    digest = bundles.eval_positions_digest()
    bundles.push_eval_positions(r2, digest)
    manifest = bundles.BundleManifest(
        bundle_id="b1", git_sha="s", git_dirty=False, archs=[], eval_positions=digest
    )
    monkeypatch.setattr(bundles, "read_manifest", lambda r2, bid: manifest)
    monkeypatch.setenv("SCZ_BUNDLE_ID", "b1")

    # The worker's copy drifts (or never existed): it is fetched at the version.
    (dirs[0] / "pos-01.gcg").write_text("stale\n")
    worker_deps.fetch_eval_positions()
    assert bundles.eval_positions_digest() == digest
    # Already current: untouched (the bucket is not even consulted).
    monkeypatch.setattr(worker_deps, "rclone", None)
    worker_deps.fetch_eval_positions()

    monkeypatch.setattr(
        bundles, "read_manifest", lambda r2, bid: bundles.BundleManifest("b0", "s", False, [])
    )
    with pytest.raises(AssertionError, match="predates"):
        worker_deps.fetch_eval_positions()


def test_a_checkout_run_worker_needs_the_checkout_datasets(datasets, monkeypatch):
    root, dirs = datasets
    monkeypatch.delenv("SCZ_BUNDLE_ID", raising=False)
    worker_deps.fetch_eval_positions()  # present: fine, no bucket involved
    import shutil

    shutil.rmtree(dirs[1])
    with pytest.raises(AssertionError, match="missing from the checkout"):
        worker_deps.fetch_eval_positions()
    assert "SCZ_BUNDLE_ID" not in os.environ


def test_the_trainer_refuses_to_start_without_its_eval_datasets(tmp_path, monkeypatch):
    """A run without its eval curves is not the run anyone asked for; on a
    rented machine it would train for hours before anyone noticed. So a
    missing dataset is a startup error, not a log line."""
    pytest.importorskip("torch")
    from scribblez.position_eval import analysis, trainer

    monkeypatch.setattr(analysis, "DEFAULT_DATASET", tmp_path / "absent")
    monkeypatch.setattr(analysis, "LARGE_DATASET", tmp_path / "absent-large")
    with pytest.raises(Exception, match="absent"):
        trainer.load_position_eval(87)
    with pytest.raises(Exception, match="absent"):
        trainer.load_position_eval_quality(87, face_up_leaves=True)
