"""Test the React dashboard launcher: port reclamation (mirrors the C++ web
server: a stale dashboard holding the port is killed before relaunch), and a
stop that returns only once its processes are gone, so a restart never races
the dashboard it replaces."""

import shutil
import subprocess
import sys
import time

import pytest
from scribblez.dashboard import react_server

# A generous ceiling on how long the listener takes to come up: it is normally
# reached in milliseconds, so waiting on the condition rather than
# sleeping a fixed span costs nothing when it holds and still fails the test
# (rather than hanging) when it does not.
_APPEAR_TIMEOUT = 5.0


def _wait_until(predicate, timeout: float) -> bool:
    """Poll `predicate` until it holds or `timeout` elapses. Returns whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_reclaim_port_kills_listener():
    if not shutil.which("lsof"):
        pytest.skip("lsof unavailable")

    # Let the holder itself pick a free port and report it back over stdout,
    # rather than picking one here and closing it -- a bind-then-close-then-
    # reopen leaves a window where another process could grab the same port
    # before the holder rebinds it.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "s.bind(('0.0.0.0',0));s.listen();"
            "print(s.getsockname()[1],flush=True);time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(holder.stdout.readline())
        assert _wait_until(
            lambda: holder.pid in react_server._listening_pids(port), _APPEAR_TIMEOUT
        )
        react_server.reclaim_port(port)
        assert react_server._listening_pids(port) == []  # free by the time it returns
    finally:
        holder.kill()


# A process that answers SIGTERM by finishing what it is doing for a second,
# as the API finishes its step in flight, and one that never exits on it.
_SLOW_TO_STOP = (
    "import signal,sys,time;"
    "signal.signal(signal.SIGTERM, lambda *a: (time.sleep(1), sys.exit(0)));"
    "print('ready',flush=True);time.sleep(30)"
)
_DEAF_TO_STOP = (
    "import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);"
    "print('ready',flush=True);time.sleep(30)"
)


def _started(code: str) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"  # its handler is installed
    return p


def test_a_stop_returns_only_once_the_process_has_exited():
    p = _started(_SLOW_TO_STOP)
    react_server._stop([p], {p.pid: "The API"})
    assert p.returncode == 0  # it finished and exited; it was not killed


def test_a_stop_kills_a_process_that_overstays(monkeypatch):
    monkeypatch.setattr(react_server, "STOP_SECONDS", 0.5)
    p = _started(_DEAF_TO_STOP)
    react_server._stop([p], {p.pid: "The API"})
    assert p.returncode == -9
