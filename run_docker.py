#!/usr/bin/env python3
"""Launch (or attach to) the Scribblez Docker container.

Mounts:
  <repo>       -> /workspace/repo   (your live source tree; build artifacts in
                                     target/ persist on the host)
  <mount-dir>  -> /workspace/mount  (Macondo checkout, lexica, future data)

The engine's and dashboard's web UIs are reached via devenv-gateway hostnames
(http://scribblez-<service>.localhost); the launcher prints the service -> URL
table at every launch (see subtrees/devenv_utils/GATEWAY.md).

Drops you into a bash shell inside the container as `devuser`, whose UID/GID
match your host user so that anything written into the bind-mounts is owned
by you on the host.
"""

from setup_common import check_setup_version, make_config
from subtrees.devenv_utils import docker_launch


def main():
    check_setup_version()
    docker_launch(make_config())


if __name__ == "__main__":
    main()
