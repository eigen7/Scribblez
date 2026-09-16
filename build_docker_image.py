#!/usr/bin/env python3
"""Build the local Scribblez Docker image from docker-setup/local/.

Only that: this is what a collaborator runs to build their own dev image for
run_docker.py. The cloud worker image is a derivative of the dev image (it
copies its C++ and NVIDIA runtime libraries), built and pushed separately by
the registry's owner with build_and_push_worker_image.py; the dashboard
refuses to deploy a bundle a published worker image cannot load
(cloud/runtime_abi.py), which is what keeps the two in step.
"""

from setup_common import make_config
from subtrees.devenv_utils import docker_build


def main():
    docker_build(make_config())


if __name__ == "__main__":
    main()
