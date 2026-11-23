"""The workload-rename migration (py/scripts/rename_workload_tags.py): a
renamed workload's tags move under its new name, with the workload and role
names each stored file and control record carries, and a renamed data/
subdirectory; a tag that cannot move refuses the whole run before anything
changes."""

import importlib.util
import json
import os

import pytest
from scribblez.dashboard.control_store import ControlStore
from scribblez.paths import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "rename_workload_tags", REPO_ROOT / "py" / "scripts" / "rename_workload_tags.py"
)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

OLD, NEW = "old_workload", "new_workload"


def _worker(pid=None) -> dict:
    return {"worker_id": "ssh-0", "role": "truth", "pid": pid}


def _tag(mount, tag: str, pid=None):
    """A finished tag of OLD: its files, and its control record."""
    root = mount / "tags" / OLD / tag
    for folder in ("params", "stats", "data/truth"):
        (root / folder).mkdir(parents=True)
    (root / "task.json").write_text(json.dumps({"workload": OLD, "tag": tag, "params": {}}))
    worker = {"worker_id": "ssh-0", "workload": OLD, "role": "truth", "tag": tag}
    (root / "params" / "ssh-0.json").write_text(json.dumps(worker))
    (root / "stats" / "ssh-0.json").write_text(json.dumps({"role": "truth", "units_total": 3}))
    (root / "data" / "truth" / "pos.json").write_text("{}")
    ControlStore(mount).put("task", f"{OLD}/{tag}", json.dumps({"workers": [_worker(pid)]}))


def _run(monkeypatch, mount, *extra) -> int:
    argv = [OLD, NEW, "--role", "truth=rollouts", "--data-dir", "truth=hasty_rollouts"]
    monkeypatch.setattr("sys.argv", ["rename", *argv, *extra, "--mount-root", str(mount)])
    return tool.main()


def test_every_tag_moves_under_the_new_name(tmp_path, monkeypatch):
    for tag in ("v1", "v2"):
        _tag(tmp_path, tag)

    assert _run(monkeypatch, tmp_path) == 0

    assert not (tmp_path / "tags" / OLD).exists()
    store = ControlStore(tmp_path)
    for tag in ("v1", "v2"):
        root = tmp_path / "tags" / NEW / tag
        assert json.loads((root / "task.json").read_text())["workload"] == NEW
        params = json.loads((root / "params" / "ssh-0.json").read_text())
        assert (params["workload"], params["role"]) == (NEW, "rollouts")
        stats = json.loads((root / "stats" / "ssh-0.json").read_text())
        assert stats == {"role": "rollouts", "units_total": 3}
        assert (root / "data" / "hasty_rollouts" / "pos.json").is_file()
        assert store.get("task", f"{OLD}/{tag}") is None
        record = json.loads(store.get("task", f"{NEW}/{tag}"))
        assert record["workers"][0]["role"] == "rollouts"


def test_a_tag_with_a_live_worker_refuses_the_run_before_anything_moves(tmp_path, monkeypatch):
    _tag(tmp_path, "v1")
    _tag(tmp_path, "v2", pid=os.getpid())

    with pytest.raises(SystemExit, match="v2: workers still alive: ssh-0"):
        _run(monkeypatch, tmp_path)

    assert (tmp_path / "tags" / OLD / "v1" / "data" / "truth").is_dir()
    assert ControlStore(tmp_path).get("task", f"{OLD}/v1") is not None


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    _tag(tmp_path, "v1")
    before = (tmp_path / "tags" / OLD / "v1" / "params" / "ssh-0.json").read_text()

    assert _run(monkeypatch, tmp_path, "--dry-run") == 0

    assert "data/truth -> data/hasty_rollouts" in capsys.readouterr().out
    assert (tmp_path / "tags" / OLD / "v1" / "params" / "ssh-0.json").read_text() == before
    assert not (tmp_path / "tags" / NEW).exists()


def test_a_queue_that_names_the_old_workload_is_refused(tmp_path, monkeypatch):
    _tag(tmp_path, "v1")
    ControlStore(tmp_path).put("queue", "", json.dumps({"entries": [{"workload": OLD}]}))

    with pytest.raises(SystemExit, match="the queue names old_workload"):
        _run(monkeypatch, tmp_path)
