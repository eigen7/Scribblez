"""Fences every test off from the live dashboard's state under
/workspace/mount/tags: a test that reaches a tag dir through the default mount
root, or runs cloud_sync, would otherwise write into (and recreate) real
tags. cloud_sync is told the redirected root (workers.cloud_sync_argv), but it
still pulls from the real bucket, so both of its launches, the long-lived
watcher and the drain's one-off pull, are stubbed; the tests of the watcher
itself restore the real method explicitly.
"""

import pytest
from scribblez.dashboard.workers import WorkerManager
from scribblez.paths import TagPaths
from scribblez.workloads import WorkloadSpec


@pytest.fixture(autouse=True)
def _isolate_tag_dirs(tmp_path_factory, monkeypatch):
    mount = tmp_path_factory.mktemp("mount")
    monkeypatch.setattr(
        WorkloadSpec,
        "paths",
        lambda self, tag, mount_root=None: TagPaths(tag, self.name, mount_root or mount),
    )
    monkeypatch.setattr(WorkerManager, "_ensure_sync", lambda self, spec, task: None)
    monkeypatch.setattr(WorkerManager, "sync_once", lambda self, spec, task: None)
