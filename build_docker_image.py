#!/usr/bin/env python3
"""Build the local Scribblez Docker image from docker-setup/local/.

The cloud worker image is built and pushed straight after, because it is a
derivative of this one: it copies this image's C++ and NVIDIA runtime
libraries, and bundles compiled here have to load there. Rebuilding one alone
is how workers end up unable to run current code, so the two move together.
"""

import sys
from pathlib import Path

from build_and_push_worker_image import build_and_push
from setup_common import make_config
from subtrees.devenv_utils import docker_build

sys.path.insert(0, str(Path(__file__).resolve().parent / "py"))
from cloud.credentials import CredentialsError  # noqa: E402


def main():
    config = make_config()
    docker_build(config)
    try:
        build_and_push(config)
    except CredentialsError as e:
        # No cloud credentials: this machine builds and runs locally and has no
        # worker image to keep in step with.
        print(f"\nSkipping the worker image: {e}")


if __name__ == "__main__":
    main()
