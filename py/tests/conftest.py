"""Keeps every test off the live bucket. Tests cannot reach the live tag trees
by omission: nothing below an entry point defaults to the real mount root, so
each test roots its WorkerManager (or TagPaths) in a scratch dir. cloud_sync
is told that root (WorkerManager.cloud_sync_argv), but it still pulls from the
real bucket, so both of its launches, the long-lived watcher and the drain's
one-off pull, are stubbed; the tests of the watcher itself restore the real
method explicitly.
"""

import pytest
from scribblez.dashboard.workers import WorkerManager


@pytest.fixture(autouse=True)
def _no_bucket_sync(monkeypatch):
    monkeypatch.setattr(WorkerManager, "_ensure_sync", lambda self, spec, task: None)
    monkeypatch.setattr(WorkerManager, "sync_once", lambda self, spec, task: None)
