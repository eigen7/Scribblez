"""Test the React dashboard launcher's port reclamation (mirrors the C++ web
server: a stale dashboard holding the port is killed before relaunch)."""

import shutil
import socket
import subprocess
import sys
import time

import pytest
from scribblez.dashboard import react_server


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
        time.sleep(1.0)
        assert holder.pid in react_server._listening_pids(port)
        react_server.reclaim_port(port)
        time.sleep(0.5)
        assert react_server._listening_pids(port) == []
    finally:
        holder.kill()
