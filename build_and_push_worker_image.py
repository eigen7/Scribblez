#!/usr/bin/env python3
"""Build and push the cloud worker Docker image from docker-setup/worker/.

Runs on the host (needs the docker CLI and the local dev image, out of which
the worker image copies the C++ and NVIDIA runtime libraries). Pushes to the
private registry repo named in <mount>/cloud/credentials.json
(registry.worker_image); `docker login` for that registry must have been run
beforehand.

Two images come out of one Dockerfile, one per runtime a role declares
(py/cloud/runtime_abi.py): the engine runtime, and the torch runtime layered
on it for the train roles, pushed under the engine image's tag with "-torch"
appended. Both contain dependencies only -- code and binaries reach workers
through R2 bundles (py/scripts/cloud_push_binaries.py) -- but those
dependencies are the dev image's, so the images are a matched set: bundles
built in a dev image whose libraries have moved cannot load on a worker image
built before they did. build_docker_image.py therefore runs this too, and
each push records what its image provides (cloud/runtime_abi.py) so the dev
container can refuse to deploy a bundle a published image cannot run.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from setup_common import make_config
from subtrees.devenv_utils import get_env_json, in_docker_container

REPO_ROOT = Path(__file__).resolve().parent
WORKER_CONTEXT = REPO_ROOT / "docker-setup" / "worker"


def _repo_py_on_path():
    if str(REPO_ROOT / "py") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "py"))


def mount_dir(config) -> Path:
    path = get_env_json(config.env_json_path).get("MOUNT_DIR")
    assert path, "MOUNT_DIR is not set; run ./setup_wizard.py first."
    return Path(path)


def load_registry(config):
    _repo_py_on_path()
    from cloud.credentials import load_credentials

    return load_credentials(mount_dir(config) / "cloud" / "credentials.json").registry


def probe_versions(image: str) -> dict[str, str]:
    """The runtime library versions `image` provides, asked of the image
    itself -- it holds no repo to import cloud/runtime_abi.py from."""
    _repo_py_on_path()
    from cloud import runtime_abi

    _, _, script = runtime_abi.probe_command()
    res = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return runtime_abi.parse_versions(res.stdout)


def dev_torch_spec(dev_image: str) -> str:
    """The pip requirement pinning torch to the dev image's version, e.g.
    "torch==2.9.0+cu129"."""
    res = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            dev_image,
            "python3",
            "-c",
            "import torch; print(torch.__version__)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return f"torch=={res.stdout.strip()}"


def build_and_push(config) -> list[str]:
    """Build both worker images from the dev image, push them, and record what
    each provides. Returns the image names."""
    _repo_py_on_path()
    from cloud import runtime_abi

    registry = load_registry(config)
    torch_spec = dev_torch_spec(config.image)
    images = []
    for runtime in runtime_abi.RUNTIMES:
        image = registry.image_for(runtime)
        for cmd in (
            ["docker", "build", "-t", image, "--target", runtime_abi.DOCKER_TARGET[runtime],
             "--build-arg", f"DEV_IMAGE={config.image}", "--build-arg", f"TORCH_SPEC={torch_spec}",
             str(WORKER_CONTEXT)],
            ["docker", "push", image],
        ):  # fmt: skip
            print(f"$ {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        versions = probe_versions(image)
        runtime_abi.write_record(mount_dir(config), runtime, image, versions)
        print(f"Pushed {image} ({runtime} runtime), providing {json.dumps(versions)}.")
        images.append(image)
    return images


def main():
    assert not in_docker_container(), "build_and_push_worker_image.py must run on the host."
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    build_and_push(make_config())


if __name__ == "__main__":
    main()
