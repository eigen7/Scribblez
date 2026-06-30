"""Launch the React training dashboard: the Python (Tornado) data API plus the Vite
dev server that serves the React app (``VITE_TOOL=dashboard``).

This mirrors how the C++ web tools launch Vite (ports injected via env vars), but
the backend here is the Python data API (``api.py``) rather than a C++ WebSocket
server -- the dashboard's data lives in Python (SQLite, torch, the engine FFI). The
existing Bokeh-served dashboard is unaffected and can run alongside this during the
migration. See docs/react_dashboard.md.
"""

import os
import subprocess
import sys
from pathlib import Path

from scribblez.instance_ports import port_offset

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

# Defaults shifted by the dev-container instance offset so parallel instances don't
# collide; chosen distinct from the C++ tools (8080/5173-5) and Bokeh (5006).
DEFAULT_API_PORT = 8090 + port_offset()
DEFAULT_DEV_PORT = 5180 + port_offset()


def launch(
    task: str,
    mount_root: str = "/workspace/mount",
    api_port: int = DEFAULT_API_PORT,
    dev_port: int = DEFAULT_DEV_PORT,
):
    """Spawn the data API and the Vite dev server; block until interrupted, then
    tear both down."""
    # fmt: off
    api = subprocess.Popen(
        [
            sys.executable, "-m", "scribblez.dashboard.api",
            "--port", str(api_port), "--mount-root", str(mount_root),
        ]
    )
    # fmt: on
    env = {
        **os.environ,
        "VITE_TOOL": "dashboard",
        "VITE_TASK": task,
        "VITE_DEV_PORT": str(dev_port),
        "VITE_API_PORT": str(api_port),
    }
    vite = subprocess.Popen(["npm", "run", "dev"], cwd=WEB_DIR, env=env)
    print(f"React dashboard ({task}) at http://localhost:{dev_port}", file=sys.stderr)
    try:
        vite.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (vite, api):
            p.terminate()
