#!/usr/bin/env python3
"""Build the local Scribblez Docker image from docker-setup/."""

from setup_common import make_config
from subtrees.devenv_utils import docker_build


def main():
    docker_build(make_config())


if __name__ == "__main__":
    main()
