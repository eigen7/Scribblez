"""Launch the Bokeh dashboard server.

The dashboard is a Bokeh server application (``app.py``) served over a single
port. It can run standalone (``py/scripts/dashboard.py``) or be spawned by
train.py so the dashboard is live while training runs.
"""

import subprocess
import sys
from pathlib import Path

from scribblez.instance_ports import port_offset

APP_PATH = Path(__file__).with_name("app.py")
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


def serve_command(port: int = DEFAULT_PORT, mount_root: str = "/workspace/mount") -> list[str]:
    """The argv that serves the dashboard app, bound for Docker port-forwarding."""
    return [
        sys.executable, "-c", _SERVE_SHIM, "serve", str(APP_PATH),
        "--port", str(port),
        "--address", "0.0.0.0",
        "--allow-websocket-origin", f"localhost:{port}",
        "--allow-websocket-origin", f"127.0.0.1:{port}",
        "--args", "--mount-root", str(mount_root),
    ]


def launch_dashboard(
    port: int = DEFAULT_PORT, mount_root: str = "/workspace/mount", tag: str | None = None
) -> subprocess.Popen | None:
    """Spawn the dashboard as a background process; None if it can't start.

    When `tag` is given, the printed URL carries `?tag=<tag>` so the dashboard
    opens with that tag's pane selected.
    """
    try:
        proc = subprocess.Popen(serve_command(port, mount_root))
    except (FileNotFoundError, OSError) as e:  # bokeh missing / exec failure
        print(f"WARNING: could not launch dashboard ({e}); is bokeh installed?", file=sys.stderr)
        return None
    url = f"http://localhost:{port}/app"
    if tag:
        url += f"?tag={tag}"
    print(f"Dashboard: {url}")
    return proc
