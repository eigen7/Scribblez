"""The tag-param migration helper: applies rename/drop/set across a tag's
task.json and per-worker params snapshots, validating against the live schema."""

import importlib.util
import json
import os

import pytest
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.paths import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "migrate_tag_params", REPO_ROOT / "py" / "scripts" / "migrate_tag_params.py"
)
mig = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mig)

_WORKLOAD = "position_eval"  # its schema no longer has mask_placement (dropped) -> a real case


def _tag_dir(mount_root, tag, worker_ids=("local-0",), extra_params=None):
    """A minimal live-tag layout: task.json plus one params snapshot per worker,
    each carrying `extra_params` on top of the workload defaults."""
    extra_params = extra_params or {}
    root = mount_root / "tags" / _WORKLOAD / tag
    (root / "params").mkdir(parents=True)
    workers = [{"worker_id": w, "role": "generate", "pid": None} for w in worker_ids]
    (root / "task.json").write_text(
        json.dumps(
            {"workload": _WORKLOAD, "tag": tag, "params": dict(extra_params), "workers": workers}
        )
    )
    for w in worker_ids:
        (root / "params" / f"{w}.json").write_text(
            json.dumps(
                {"worker_id": w, "workload": _WORKLOAD, "tag": tag, "params": dict(extra_params)}
            )
        )
    return root


def _all_param_blocks(root):
    return [json.loads(f.read_text())["params"] for f in mig.param_files(root)]


def test_apply_ops_order_and_change_flag():
    params = {"old": 1, "keep": 2, "stale": 3}
    out, changed = mig.apply_ops(params, {"old": "new"}, ["stale"], {"added": 9})
    assert out == {"new": 1, "keep": 2, "added": 9}
    assert changed
    # A no-op set (value already present) reports no change.
    _, changed2 = mig.apply_ops({"added": 9}, {}, [], {"added": 9})
    assert not changed2


def test_drop_clears_every_stored_copy(tmp_path):
    root = _tag_dir(
        tmp_path, "t", worker_ids=("local-0", "ssh-1"), extra_params={"mask_placement": True}
    )
    spec = workloads.get(_WORKLOAD)
    changed = mig.migrate_tag(spec, "t", str(tmp_path), {}, ["mask_placement"], {}, dry_run=False)
    assert changed == 3  # task.json + two snapshots
    for block in _all_param_blocks(root):
        assert "mask_placement" not in block
        params_mod.validate(spec.params_cls, block)  # loads again


def test_dry_run_writes_nothing(tmp_path):
    root = _tag_dir(tmp_path, "t", extra_params={"mask_placement": True})
    spec = workloads.get(_WORKLOAD)
    changed = mig.migrate_tag(spec, "t", str(tmp_path), {}, ["mask_placement"], {}, dry_run=True)
    assert changed == 2
    for block in _all_param_blocks(root):
        assert "mask_placement" in block  # untouched on disk


def test_already_clean_is_idempotent(tmp_path):
    _tag_dir(tmp_path, "t", extra_params={"num_blocks": 8})
    spec = workloads.get(_WORKLOAD)
    assert mig.migrate_tag(spec, "t", str(tmp_path), {}, ["mask_placement"], {}, dry_run=False) == 0


def test_set_value_is_json_typed_then_coerced(tmp_path):
    root = _tag_dir(tmp_path, "t")
    spec = workloads.get(_WORKLOAD)
    sets = {"num_blocks": mig._json_value("8"), "face_up_leaves": mig._json_value("true")}
    mig.migrate_tag(spec, "t", str(tmp_path), {}, [], sets, dry_run=False)
    block = _all_param_blocks(root)[0]
    assert block["num_blocks"] == 8 and block["face_up_leaves"] is True


def test_refuses_a_result_that_still_fails_validation(tmp_path):
    _tag_dir(tmp_path, "t", extra_params={"mask_placement": True})
    spec = workloads.get(_WORKLOAD)
    with pytest.raises(SystemExit, match="still invalid"):
        # A rename to an unknown key -> the migration must refuse to write it.
        mig.migrate_tag(
            spec, "t", str(tmp_path), {"mask_placement": "not_a_param"}, [], {}, dry_run=False
        )


def test_refuses_while_a_worker_is_alive(tmp_path):
    root = tmp_path / "tags" / _WORKLOAD / "t"
    (root / "params").mkdir(parents=True)
    (root / "task.json").write_text(
        json.dumps(
            {
                "workload": _WORKLOAD,
                "tag": "t",
                "params": {"mask_placement": True},
                "workers": [{"worker_id": "local-0", "role": "generate", "pid": os.getpid()}],
            }
        )
    )
    spec = workloads.get(_WORKLOAD)
    with pytest.raises(SystemExit, match="still running"):
        mig.migrate_tag(spec, "t", str(tmp_path), {}, ["mask_placement"], {}, dry_run=False)
