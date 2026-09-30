"""The dashboard's control lock (api.acquire_control_lock): one dashboard per
mount, and a restart that begins while the previous one is still exiting
waits for it rather than giving up."""

import fcntl
import threading

import pytest
from scribblez.dashboard import api


@pytest.fixture
def lock_state(monkeypatch):
    """Short waits, and the lock this process takes released afterwards."""
    monkeypatch.setattr(api, "LOCK_POLL_SECONDS", 0.05)
    yield
    if api._CONTROL_LOCK is not None:
        api._CONTROL_LOCK.close()
        api._CONTROL_LOCK = None


def _held(mount) -> object:
    """The lock taken by someone else: another open file description."""
    fd = open(mount / ".dashboard.lock", "a+")
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_a_restart_waits_for_the_previous_dashboard_to_exit(tmp_path, lock_state):
    previous = _held(tmp_path)
    threading.Timer(0.3, previous.close).start()  # it finishes exiting
    api.acquire_control_lock(str(tmp_path))
    assert api._CONTROL_LOCK is not None


def test_a_dashboard_that_stays_is_refused(tmp_path, lock_state, monkeypatch):
    monkeypatch.setattr(api, "LOCK_WAIT_SECONDS", 0.2)
    other = _held(tmp_path)
    try:
        with pytest.raises(SystemExit, match="still managing"):
            api.acquire_control_lock(str(tmp_path))
    finally:
        other.close()
