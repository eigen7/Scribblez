"""Launch a per-task Bokeh dashboard server.

Each model has its own Bokeh app (``app_<task>.py`` in this package, sharing the
shell in ``shell.py``); a trainer spawns the one for its task so the dashboard is
live while training runs, or it can be served standalone via
``py/scripts/dashboard.py``.
"""

import subprocess
import sys
from pathlib import Path

from scribblez.instance_ports import port_offset

# The per-task app modules `bokeh serve` runs (the URL path is the file's stem).
POST_MOVE_VALUE_APP = "app_post_move_value.py"
MAX_MOVE_PER_LANE_APP = "app_max_move_per_lane.py"

# Base 5006, shifted by the dev-container instance offset so parallel instances
# don't collide. Consumed as the argparse default by the dashboard/train CLIs.
DEFAULT_PORT = 5006 + port_offset()


# Run `bokeh serve` through a tiny `python -c` shim that first sets logging's
# millisecond separator to a period (Bokeh prints '...:09,812'; we want
# '...:09.812'). `default_msec_format` is a public logging.Formatter attribute, so
# setting it before Bokeh configures logging changes every line, startup included.
# Using -c (not -m bokeh) keeps it working regardless of the `bokeh` console script.
_SERVE_SHIM = (
    "import sys, logging; "
    "logging.Formatter.default_msec_format = '%s.%03d'; "
    "from bokeh.command.bootstrap import main; "
    "main(['bokeh', *sys.argv[1:]])"
)


def serve_command(
    port: int = DEFAULT_PORT,
    mount_root: str = "/workspace/mount",
    app: str = POST_MOVE_VALUE_APP,
) -> list[str]:
    """The argv that serves dashboard app `app`, bound for Docker port-forwarding."""
    app_path = Path(__file__).with_name(app)
    # fmt: off
    return [
        sys.executable, "-c", _SERVE_SHIM, "serve", str(app_path),
        "--port", str(port),
        "--address", "0.0.0.0",
        "--allow-websocket-origin", f"localhost:{port}",
        "--allow-websocket-origin", f"127.0.0.1:{port}",
        "--args", "--mount-root", str(mount_root),
    ]
    # fmt: on


def launch_dashboard(
    port: int = DEFAULT_PORT,
    mount_root: str = "/workspace/mount",
    tag: str | None = None,
    app: str = POST_MOVE_VALUE_APP,
) -> subprocess.Popen | None:
    """Spawn dashboard app `app` as a background process; None if it can't start.

    When `tag` is given, the printed URL carries `?tag=<tag>` so the dashboard
    opens with that tag's pane selected.
    """
    try:
        proc = subprocess.Popen(serve_command(port, mount_root, app))
    except (FileNotFoundError, OSError) as e:  # bokeh missing / exec failure
        print(f"WARNING: could not launch dashboard ({e}); is bokeh installed?", file=sys.stderr)
        return None
    url = f"http://localhost:{port}/{Path(app).stem}"
    if tag:
        url += f"?tag={tag}"
    print(f"Dashboard: {url}")
    return proc
