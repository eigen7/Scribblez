#!/usr/bin/env python3
"""Build the local Scribblez Docker image from docker-setup/."""

import argparse
import os
import subprocess
import sys

from setup_common import LOCAL_DOCKER_IMAGE, in_docker_container


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-l", "--local-docker-image", default=LOCAL_DOCKER_IMAGE,
        help="local image tag (default: %(default)s)",
    )
    return parser.parse_args()


def docker_build(image: str) -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Building docker image {image}...")
    cmd = ["docker", "build", "-t", image, "docker-setup/"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to build docker image {image}.")
        return 1
    print(f"Successfully built docker image {image}.")
    return 0


def main() -> None:
    assert not in_docker_container(), \
        "build_docker_image.py should not be run inside a container."
    args = get_args()
    sys.exit(docker_build(args.local_docker_image))


if __name__ == "__main__":
    main()
