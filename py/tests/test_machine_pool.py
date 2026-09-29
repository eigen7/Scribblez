"""The machine pool (dashboard/pool.py) and the worker manager's use of it.

What must hold:
- a pool machine survives a save/load round trip;
- a leased machine's name resolves for the leasing task's slots and for no
  other task, since the pool rather than the task owns it;
- the busy rule counts slots that are alive or gated, and slots meant to run
  only within a grace period, across the spellings tags use for one host.
"""

import pytest
from scribblez.dashboard import pool as pool_mod
from scribblez.dashboard import tasks
from scribblez.dashboard import workers as workers_mod
from scribblez.dashboard.pool import Hardware, Lease, PoolMachine
from scribblez.dashboard.workers import WorkerManager, _key

L4_REPORT = "8\n23034\n"


class _FakeSshMachine:
    """Answers the hardware query with REPORT and every container probe with
    `state`; remembers the host of every link built."""

    report = L4_REPORT
    state = "stopped"
    built: list[str] = []

    def __init__(self, host, identity_file=None, known_hosts_file=None):
        self.host = host
        _FakeSshMachine.built.append(host)

    def hardware_report(self) -> str:
        return self.report

    def container_state(self, name: str) -> str:
        return self.state


@pytest.fixture
def pooled(tmp_path, monkeypatch):
    """A WorkerManager rooted at tmp_path, with ssh faked and the task
    listing limited to what the test registers."""
    monkeypatch.setattr(workers_mod, "SshMachine", _FakeSshMachine)
    monkeypatch.setattr(_FakeSshMachine, "built", [])
    monkeypatch.setattr(pool_mod, "local_hardware", lambda: Hardware(24, 1, 16.0))
    # ssh -G is config evaluation only, but keep the test off the real config.
    monkeypatch.setattr(pool_mod, "canonical_host", lambda h: h.split("@", 1)[-1].lower())
    manager = WorkerManager(tmp_path)
    listed: list = []
    monkeypatch.setattr(manager, "all_tasks", lambda: iter(listed))
    return manager, listed


def _task(tag: str) -> tasks.TaskRecord:
    return tasks.TaskRecord(workload="position_eval", tag=tag, params={}, created_at=0.0)


def _slot(worker_id: str, kind: str, desired: str, host: str | None = None, machine=None):
    return tasks.WorkerRecord(
        worker_id=worker_id, role="generate", kind=kind, desired_state=desired,
        host=host, machine=machine,
    )  # fmt: skip


def test_parse_hardware():
    assert pool_mod.parse_hardware(L4_REPORT) == Hardware(8, 1, 23034 / 1024)
    assert pool_mod.parse_hardware("16\n") == Hardware(16, 0, 0.0)
    # Two GPUs of different sizes: placement must fit the smaller.
    assert pool_mod.parse_hardware("32\n24000\n16000\n").gpu_memory_gb == 16000 / 1024


def test_a_pool_round_trips_through_pool_json(pooled):
    manager, _ = pooled
    manager.add_pool_machine("asus", "dshin@asus-laptop", aliases=["asus-laptop"])
    pool = manager.pool_store.load()
    pool.machine("asus").lease = Lease("position_eval", "t", "running", 1.0)
    manager.pool_store.save(pool)
    m = pool_mod.pool_store(manager.mount_root).load().machine("asus")  # a fresh process
    assert m.kind == "ssh" and m.machine.host == "dshin@asus-laptop"
    assert m.hardware == Hardware(8, 1, 23034 / 1024) and m.machine.gpu_count == 1
    assert m.aliases == ["asus-laptop"]
    assert m.lease == Lease("position_eval", "t", "running", 1.0)


def test_adding_probes_and_refuses_duplicates(pooled):
    manager, _ = pooled
    local = manager.add_pool_machine("localhost")
    assert local.kind == "local" and local.hardware == Hardware(24, 1, 16.0)
    with pytest.raises(AssertionError, match="already pooled"):
        manager.add_pool_machine("again")
    with pytest.raises(AssertionError, match="exists"):
        manager.add_pool_machine("localhost", "x@y")


def test_a_leased_machine_cannot_be_removed(pooled):
    manager, _ = pooled
    manager.add_pool_machine("asus", "asus-laptop")
    pool = manager.pool_store.load()
    pool.machine("asus").lease = Lease("position_eval", "t", "running", 1.0)
    with pytest.raises(AssertionError, match="leased"):
        manager.remove_pool_machine("asus")
    pool.machine("asus").lease = None
    manager.remove_pool_machine("asus")
    assert manager.pool_store.load().find("asus") is None


def test_edit_changes_only_operator_fields(pooled):
    manager, _ = pooled
    manager.add_pool_machine("asus", "asus-laptop")
    manager.edit_pool_machine("asus", generator_threads=6, aliases=["a"])
    m = manager.pool_store.load().machine("asus")
    assert (m.generator_threads, m.aliases) == (6, ["a"])
    with pytest.raises(AssertionError, match="not editable"):
        manager.edit_pool_machine("asus", hardware=None)


def test_a_leased_name_resolves_only_for_the_leasing_task(pooled):
    manager, _ = pooled
    manager.add_pool_machine("asus", "dshin@asus-laptop")
    pool = manager.pool_store.load()
    pool.machine("asus").lease = Lease("position_eval", "leaser", "running", 1.0)
    leaser, other = _task("leaser"), _task("other")
    assert manager._machine_record(leaser, "asus").host == "dshin@asus-laptop"
    with pytest.raises(KeyError):
        manager._machine_record(other, "asus")
    w = _slot("ssh-0", "ssh", "running", machine="asus")
    leaser.workers.append(w)
    manager._ssh_machine(leaser, w)
    assert _FakeSshMachine.built[-1] == "dshin@asus-laptop"
    # A task's own machine of the same name wins: it is the task's to name.
    leaser.machines.append(tasks.MachineRecord(name="asus", provider="manual", host="elsewhere"))
    assert manager._machine_record(leaser, "asus").host == "elsewhere"


def test_the_leasing_tasks_status_lists_the_pool_machine(pooled, monkeypatch):
    manager, _ = pooled
    manager.add_pool_machine("asus", "asus-laptop")
    manager.pool_store.load().machine("asus").lease = Lease("position_eval", "t", "running", 1.0)
    from scribblez.workloads.position_eval import SPEC

    (info,) = manager.machine_status(SPEC, _task("t"))
    assert info["name"] == "asus" and info["pool"] is True
    assert manager.machine_status(SPEC, _task("unrelated")) == []


def _occupied(manager, listed, name) -> dict:
    return next(m for m in manager.pool_status() if m["name"] == name)


def test_busy_counts_running_or_alive_slots_under_any_spelling(pooled):
    manager, listed = pooled
    from scribblez.workloads.position_eval import SPEC

    manager.add_pool_machine("asus", "dshin@asus-laptop", aliases=["asus-laptop"])
    manager.add_pool_machine("localhost")
    t = _task("hand")
    listed.append((SPEC, t))
    assert _occupied(manager, listed, "asus")["state"] == "free"

    # Paused slots, the old tags' usual leftovers, hold nothing.
    t.workers += [
        _slot("ssh-0", "ssh", "paused", host="asus-laptop"),
        _slot("local-0", "local", "paused"),
    ]
    assert _occupied(manager, listed, "asus")["state"] == "free"
    assert _occupied(manager, listed, "localhost")["state"] == "free"

    # One that wants to run holds its machine, spelled either way.
    t.workers[0].desired_state = "running"
    t.workers[1].desired_state = "running"
    assert _occupied(manager, listed, "asus")["occupants"] == ["position_eval/hand/ssh-0"]
    assert _occupied(manager, listed, "localhost")["occupants"] == ["position_eval/hand/local-0"]

    # A paused slot still seen alive (it has not wound down) holds it too.
    t.workers[0].desired_state = "paused"
    manager._probes[_key(SPEC, "hand", "ssh-0")] = ("running", 0.0)
    assert _occupied(manager, listed, "asus")["state"] == "busy"


def test_the_leasing_tags_own_slots_do_not_make_it_busy(pooled):
    manager, listed = pooled
    from scribblez.workloads.position_eval import SPEC

    manager.add_pool_machine("asus", "asus-laptop")
    manager.pool_store.load().machine("asus").lease = Lease(
        "position_eval", "placed", "running", 1.0
    )
    placed = _task("placed")
    placed.workers.append(_slot("ssh-0", "ssh", "running", machine="asus"))
    listed.append((SPEC, placed))
    status = _occupied(manager, listed, "asus")
    assert status["state"] == "leased" and status["occupants"] == []


def test_canonical_host_strips_the_user_and_lowercases():
    pool_mod._canonical.clear()
    assert pool_mod.canonical_host("someone@No-Such-Host-Scz") == "no-such-host-scz"
    assert pool_mod.canonical_host("No-Such-Host-Scz") == "no-such-host-scz"


def test_pool_machine_defaults():
    m = PoolMachine(name="x", kind="local")
    assert m.lease is None and m.aliases == [] and m.hardware == Hardware()


def test_a_slot_meant_to_run_but_long_dead_does_not_hold_its_machine(pooled, monkeypatch):
    """An `exited` slot (meant to run, not alive) holds its machine only for a
    grace period, which covers a restart; one that never comes back, like a
    month-old crashed generator, stops counting. A gated slot keeps counting:
    its gate lifts."""
    manager, listed = pooled
    from scribblez.workloads.position_eval import SPEC

    manager.add_pool_machine("localhost")
    t = _task("stale")
    t.workers.append(_slot("local-0", "local", "running"))  # no pid: not alive
    listed.append((SPEC, t))
    assert _occupied(manager, listed, "localhost")["state"] == "busy"  # within the grace

    monkeypatch.setattr(workers_mod, "DEAD_SLOT_GRACE_SECONDS", 0.0)
    assert _occupied(manager, listed, "localhost")["state"] == "free"

    t.gates["generate"] = "waiting for the trainer"
    assert _occupied(manager, listed, "localhost")["state"] == "busy"


def test_nvidia_smi_failure_text_is_no_gpu():
    """nvidia-smi prints its failures on stdout; a machine whose driver is not
    loaded has no usable GPU, and the probe must say so, not crash."""
    report = "8\nNVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
    assert pool_mod.parse_hardware(report) == Hardware(8, 0, 0.0)


def test_a_pool_machine_a_slot_names_cannot_be_removed(pooled):
    manager, listed = pooled
    from scribblez.workloads.position_eval import SPEC

    manager.add_pool_machine("asus", "asus-laptop")
    t = _task("stale")
    t.workers.append(_slot("ssh-0", "ssh", "paused", machine="asus"))
    listed.append((SPEC, t))
    with pytest.raises(AssertionError, match="slots still name asus: position_eval/stale/ssh-0"):
        manager.remove_pool_machine("asus")


def test_a_detected_arch_is_saved_to_the_pool(pooled, monkeypatch):
    """A leased pool machine's arch, detected at its first slot start, lands in
    pool.json rather than only in this process's copy."""
    manager, _ = pooled
    from scribblez.workloads.position_eval import SPEC

    manager.add_pool_machine("asus", "asus-laptop")
    pool = manager.pool_store.load()
    pool.machine("asus").lease = Lease("position_eval", "t", "running", 1.0)
    manager.pool_store.save(pool)
    monkeypatch.setattr(_FakeSshMachine, "pull_image", lambda self, image: None, raising=False)
    monkeypatch.setattr(_FakeSshMachine, "detect_arch", lambda self, image: "znver2", raising=False)
    registry = type("R", (), {"image_for": staticmethod(lambda runtime: "img")})
    monkeypatch.setattr(manager, "_creds", lambda: type("C", (), {"registry": registry})())
    t = _task("t")
    w = _slot("ssh-0", "ssh", "running", machine="asus")
    t.workers.append(w)
    assert manager._slot_arch(SPEC, t, w) == "znver2"
    fresh = pool_mod.pool_store(manager.mount_root)  # a fresh process
    assert fresh.load().machine("asus").machine.arch == "znver2"
