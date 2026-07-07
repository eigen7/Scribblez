"""Launch the React training dashboard: the Python (Tornado) data API plus the Vite
dev server that serves the React app (``VITE_TOOL=dashboard``).

This mirrors how the C++ web tools launch Vite (ports injected via env vars), but
the backend here is the Python data API (``api.py``) rather than a C++ WebSocket
server -- the dashboard's data lives in Python (SQLite, torch, the engine FFI). The
existing Bokeh-served dashboard is unaffected and can run alongside this during the
migration. See docs/react_dashboard.md.
"""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from scribblez.instance_ports import port_offset

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

# Defaults shifted by the dev-container instance offset so parallel instances don't
# collide; chosen distinct from the C++ tools (8080/5173-5) and Bokeh (5006).
DEFAULT_API_PORT = 8090 + port_offset()
DEFAULT_DEV_PORT = 5180 + port_offset()


def _listening_pids(port: int) -> list[int]:
    """PIDs LISTENing on `port` (TCP), via lsof. Client sockets are excluded
    (`-sTCP:LISTEN`) so an unrelated client connection is never matched."""
    lsof = shutil.which("lsof")
    if not lsof:
        return []
    try:
        out = subprocess.run(
            [lsof, "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def reclaim_port(port: int):
    """Free `port` by killing any process LISTENing on it. Mirrors the C++ web
    server, which reclaims a stale dev-server port: a leftover dashboard from a
    prior run otherwise makes Vite (or the API) fail to bind on relaunch."""
    for pid in _listening_pids(port):
        print(f"  Port {port} is in use by pid {pid}; reclaiming it.", file=sys.stderr)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _dashboard_banner(url: str) -> str:
    """A prominent, blank-line-padded banner pointing at the dashboard URL, so it
    stands out from the surrounding training and npm output. Bold cyan on a
    terminal; plain text when stderr is redirected to a file (no stray escapes)."""
    line = f"Dashboard: {url}"
    if sys.stderr.isatty():
        line = f"\033[1;36m{line}\033[0m"
    return f"\n{line}\n"


def spawn(
    task: str | None,
    mount_root: str = "/workspace/mount",
    api_port: int = DEFAULT_API_PORT,
    dev_port: int = DEFAULT_DEV_PORT,
    tag: str | None = None,
) -> list[subprocess.Popen]:
    """Spawn the data API and the Vite dev server in the background; return their
    processes (so a caller can terminate them on exit). Non-blocking, so a trainer
    can launch the dashboard alongside training. Any process already holding the
    API or Vite port is reclaimed first (a stale dashboard from a prior run). When
    `tag` is given, the printed URL carries `?tag=<tag>` so the dashboard opens on
    that run."""
    reclaim_port(api_port)
    reclaim_port(dev_port)
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scribblez.dashboard.api",
            "--port",
            str(api_port),
            "--mount-root",
            str(mount_root),
        ]
    )
    # fmt: on
    # With a task, the React app renders that training-task view (the trainers'
    # auto-spawned dashboards); with task=None it renders the master dashboard.
    env = {
        **os.environ,
        "VITE_TOOL": "dashboard",
        **({"VITE_TASK": task} if task else {}),
        "VITE_DEV_PORT": str(dev_port),
        "VITE_API_PORT": str(api_port),
    }
    # Vite's own startup chatter (the npm "> dev" lines, the ready banner, and a
    # transient "/api proxy ECONNREFUSED" while the API port is still binding) is
    # discarded: it's noise, and its bare "Local: http://.../" URL tempts a click
    # on a tag-less page. The one URL worth clicking is printed below instead.
    vite = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{dev_port}"
    if tag:
        url += f"/?tag={quote(tag)}"
    print(_dashboard_banner(url), file=sys.stderr)
    return [api, vite]


def launch(
    task: str | None,
    mount_root: str = "/workspace/mount",
    api_port: int = DEFAULT_API_PORT,
    dev_port: int = DEFAULT_DEV_PORT,
    tag: str | None = None,
):
    """Spawn the dashboard and block until interrupted, then tear it down (the CLI
    entry point)."""
    procs = spawn(task, mount_root, api_port, dev_port, tag)
    try:
        procs[-1].wait()  # the Vite process
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
