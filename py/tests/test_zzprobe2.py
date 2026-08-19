import pytest
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.workers import WorkerManager
from tests.test_worker_manager import (  # noqa: F401
    _FakeSshMachine, _fail, manager, spec, task,
)


def test_probe_stale_undelivered_survives_slot_removal(manager, spec, task, monkeypatch):
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "state", "stopped")
    w = manager.add_ssh(spec, task, "generate", host="user@laptop", threads=None)
    key = workers_mod._key(spec, task.tag, w.worker_id)
    manager._undelivered[key] = 900
    monkeypatch.setattr(_FakeSshMachine, "remove_container", lambda self, n: None, raising=False)
    manager.remove_worker(spec, task, w.worker_id)
    w2 = manager.add_ssh(spec, task, "generate", host="user@other", threads=None)
    print("reused id:", w2.worker_id)
    (info,) = manager.worker_status(spec, task, observe=True)
    print("fresh slot status:", {k: info[k] for k in info if k in ("state", "undelivered")})
