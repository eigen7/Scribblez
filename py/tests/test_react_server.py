"""Test the React dashboard launcher's port reclamation (mirrors the C++ web
server: a stale dashboard holding the port is killed before relaunch)."""

import shutil
import socket
import subprocess
import sys
import time

import pytest
from scribblez.dashboard import react_server

# Generous ceilings on how long the listener takes to come up and to die: both
# are normally reached in milliseconds, so waiting on the condition rather than
# sleeping a fixed span costs nothing when it holds and still fails the test
# (rather than hanging) when it does not.
_APPEAR_TIMEOUT = 5.0
_VANISH_TIMEOUT = 5.0


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

    # Pick a free port, then hold it with a subprocess listener.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            f"s.bind(('0.0.0.0',{port}));s.listen();time.sleep(30)",
        ]
    )
    try:
        assert _wait_until(
            lambda: holder.pid in react_server._listening_pids(port), _APPEAR_TIMEOUT
        )
        react_server.reclaim_port(port)
        assert _wait_until(lambda: react_server._listening_pids(port) == [], _VANISH_TIMEOUT)
    finally:
        holder.kill()
