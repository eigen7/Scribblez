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
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from scribblez.paths import REPO_ROOT
from scribblez.service_urls import service_url

WEB_DIR = REPO_ROOT / "web"

# Distinct from the C++ web tools' ports. The browser opens the Vite dev server,
# which the gateway routes as scribblez-dash.localhost; the loopback-only API is
# reached through Vite's proxy.
DEFAULT_API_PORT = 8090
DEFAULT_DEV_PORT = 5180

# How long a killed listener may take to release its port.
PORT_RELEASE_SECONDS = 10
# How long a stopping dashboard's processes get to exit before they are
# killed. The API finishes the step in flight (an ssh command, an upload) and
# SIGTERMs its local workers first; killing it sooner would lose that step.
STOP_SECONDS = 120


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
    # A kill returns before the port is free, and a server started meanwhile
    # would fail to bind.
    deadline = time.monotonic() + PORT_RELEASE_SECONDS
    while _listening_pids(port) and time.monotonic() < deadline:
        time.sleep(0.2)


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
    # Vite's output goes to a log rather than the terminal: it is startup
    # noise (including a transient proxy ECONNREFUSED while the API binds), and
    # its bare "Local:" URL would tempt a click on a tag-less page. The URL
    # worth clicking is printed below; the log's tail is printed if Vite exits.
    vite = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
        stdout=_vite_log(dev_port),
        stderr=subprocess.STDOUT,
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
    """The CLI entry point: run the dashboard until interrupted (Ctrl-C or
    SIGTERM) or until either process exits, then stop both and return only
    once they are gone, so a restart never races the dashboard it replaces."""
    signal.signal(signal.SIGTERM, _interrupt)
    api, vite = spawn(mount_root, api_port, dev_port, workload, tag)
    names = {api.pid: "The API", vite.pid: "Vite"}
    try:
        while (ended := next((p for p in (api, vite) if p.poll() is not None), None)) is None:
            time.sleep(0.5)
        print(f"\n{names[ended.pid]} exited (code {ended.returncode}).", file=sys.stderr)
        if ended is vite:
            print(_tail(_vite_log_path(dev_port)), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        _stop([api, vite], names)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _stop(procs: list[subprocess.Popen], names: dict[int, str]):
    """Terminate each process and wait for it to exit, killing one that takes
    longer than STOP_SECONDS."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print(f"Waiting for {names[p.pid]} (pid {p.pid}) to exit...", file=sys.stderr)
            try:
                p.wait(timeout=STOP_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"Killing {names[p.pid]} (pid {p.pid}).", file=sys.stderr)
                p.kill()
                p.wait()


def _vite_log_path(dev_port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"scribblez-dashboard-vite-{dev_port}.log"


def _vite_log(dev_port: int):
    return open(_vite_log_path(dev_port), "w")


def _tail(path: Path, lines: int = 20) -> str:
    text = path.read_text(errors="replace") if path.exists() else ""
    return "\n".join(text.splitlines()[-lines:]) or f"({path} is empty)"
