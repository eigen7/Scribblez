#!/usr/bin/env python3
"""Build and push the cloud worker Docker image from docker-setup/worker/.

Runs on the host (needs the docker CLI and the local dev image, out of which
the worker image copies the NVIDIA runtime libraries). Pushes to the private
registry repo named in <mount>/cloud/credentials.json (registry.worker_image);
`docker login` for that registry must have been run beforehand.

The worker image contains dependencies only -- code and binaries reach workers
through R2 bundles (py/scripts/cloud_push_binaries.py) -- so this needs
rerunning only when docker-setup/worker/ changes.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from setup_common import make_config
from submodules.devenv_utils import get_env_json, in_docker_container

REPO_ROOT = Path(__file__).resolve().parent
WORKER_CONTEXT = REPO_ROOT / "docker-setup" / "worker"


def load_worker_image_name(config) -> str:
    sys.path.insert(0, str(REPO_ROOT / "py"))
    from cloud.credentials import load_credentials

    mount_dir = get_env_json(config.env_json_path).get("MOUNT_DIR")
    assert mount_dir, "MOUNT_DIR is not set; run ./setup_wizard.py first."
    creds = load_credentials(Path(mount_dir) / "cloud" / "credentials.json")
    return creds.registry.worker_image


def main():
    assert not in_docker_container(), "build_and_push_worker_image.py must run on the host."
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    config = make_config()
    image = load_worker_image_name(config)
    for cmd in (
        [
            "docker",
            "build",
            "-t",
            image,
            "--build-arg",
            f"DEV_IMAGE={config.image}",
            str(WORKER_CONTEXT),
        ],
        ["docker", "push", image],
    ):
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print(f"Pushed {image}.")


if __name__ == "__main__":
    main()
