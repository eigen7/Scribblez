#!/usr/bin/env python3
"""Launch (or attach to) the Scribblez Docker container.

Mounts:
  <repo>       -> /workspace/repo   (your live source tree; build artifacts in
                                     build/ persist on the host)
  <mount-dir>  -> /workspace/mount  (Macondo checkout, lexica, future data)

Forwards the ports the engine's web UI uses (Vite + WebSocket).

Drops you into a bash shell inside the container as `devuser`, whose UID/GID
match your host user so that anything written into the bind-mounts is owned
by you on the host.
"""

from setup_common import make_config
from subtrees.devenv_utils import docker_launch


def main() -> None:
    docker_launch(make_config())


if __name__ == "__main__":
    main()
