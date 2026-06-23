"""Launch the Bokeh dashboard server.

The dashboard is a Bokeh server application (``app.py``) served over a single
port. It can run standalone (``py/scripts/dashboard.py``) or be spawned by
train.py so the dashboard is live while training runs.
"""

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
DEFAULT_PORT = 5006


def serve_command(port: int = DEFAULT_PORT, mount_root: str = "/workspace/mount") -> list[str]:
    """The `bokeh serve` argv for the dashboard app, bound for Docker port-forwarding.

    Invokes Bokeh via ``python -m bokeh`` so it works regardless of whether the
    ``bokeh`` console script is on PATH.
    """
    return [
        sys.executable, "-m", "bokeh", "serve", str(APP_PATH),
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
