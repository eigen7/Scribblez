"""Per-instance network-port offset for parallel dev containers.

run_docker.py exports DEVENV_INSTANCE_PORT_OFFSET (= instance_port_stride *
INSTANCE) into the container so multiple instances launched from sibling repo
clones bind non-colliding ports. Add port_offset() to a *default* port only; an
explicitly requested port is used verbatim.
"""

import os

ENV_VAR = "DEVENV_INSTANCE_PORT_OFFSET"


def port_offset() -> int:
    """The configured port offset, or 0 if unset/empty/not a non-negative int."""
    raw = os.environ.get(ENV_VAR, "")
    try:
        offset = int(raw)
    except ValueError:
        return 0
    return offset if offset >= 0 else 0
