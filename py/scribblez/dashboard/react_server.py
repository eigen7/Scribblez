"""Launch the master dashboard: the Tornado API (api.py) plus the Vite dev
server that serves the React app (``VITE_TOOL=dashboard``).

Vite is launched the way the C++ web tools launch it, with ports passed in env
vars, but the backend is Python rather than a C++ WebSocket server because the
dashboard's data lives in Python (SQLite, torch, the engine FFI). See
docs/master_dashboard.md and docs/react_dashboard.md.
"""

import os
import shutil
import signal
import subprocess
import sys
from urllib.parse import quote

from scribblez.paths import REPO_ROOT
from scribblez.service_urls import service_url

WEB_DIR = REPO_ROOT / "web"

# Distinct from the C++ web tools' ports. The browser opens the Vite dev server,
# which the gateway routes as scribblez-dash.localhost; the loopback-only API is
# reached through Vite's proxy.
DEFAULT_API_PORT = 8090
DEFAULT_DEV_PORT = 5180


def _listening_pids(port: int) -> list[int]:
    """PIDs listening on TCP `port`. Only listeners: an unrelated client
    connection to the port must not be matched."""
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
    """Free `port` by killing whatever listens on it, as the C++ web server does:
    a leftover dashboard would otherwise make Vite or the API fail to bind."""
    for pid in _listening_pids(port):
        print(f"  Port {port} is in use by pid {pid}; reclaiming it.", file=sys.stderr)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _dashboard_banner(url: str) -> str:
    """The dashboard URL, set off from the surrounding output. Bold cyan on a
    terminal; plain when stderr is redirected, so logs get no escape codes."""
    line = f"Dashboard: {url}"
    if sys.stderr.isatty():
        line = f"\033[1;36m{line}\033[0m"
    return f"\n{line}\n"


def spawn(
    mount_root: str = "/workspace/mount",
    api_port: int = DEFAULT_API_PORT,
    dev_port: int = DEFAULT_DEV_PORT,
    workload: str | None = None,
    tag: str | None = None,
) -> list[subprocess.Popen]:
    """Spawn the API and the Vite dev server in the background and return their
    processes for the caller to terminate. The printed URL carries `workload`
    and `tag`, when given, so the dashboard opens on that task."""
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
    env = {
        **os.environ,
        "VITE_TOOL": "dashboard",
        "VITE_DEV_PORT": str(dev_port),
        "VITE_API_PORT": str(api_port),
    }
    # Vite's output is discarded: it is startup noise (including a transient
    # proxy ECONNREFUSED while the API binds), and its bare "Local:" URL would
    # tempt a click on a tag-less page. The URL worth clicking is printed below.
    vite = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = service_url("dash", dev_port, DEFAULT_DEV_PORT)
    query = [
        *(["workload=" + quote(workload)] if workload else []),
        *(["tag=" + quote(tag)] if tag else []),
    ]
    if query:
        url += "/?" + "&".join(query)
    print(_dashboard_banner(url), file=sys.stderr)
    return [api, vite]


def launch(
    mount_root: str = "/workspace/mount",
    api_port: int = DEFAULT_API_PORT,
    dev_port: int = DEFAULT_DEV_PORT,
    workload: str | None = None,
    tag: str | None = None,
):
    """The CLI entry point: run the dashboard until interrupted."""
    procs = spawn(mount_root, api_port, dev_port, workload, tag)
    try:
        procs[-1].wait()  # the Vite process
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
