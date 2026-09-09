"""Tests for the cgroup-aware default thread count (scribblez.hardware)."""

from scribblez import hardware


def _v2(tmp_path, monkeypatch, contents: str):
    path = tmp_path / "cpu.max"
    path.write_text(contents)
    monkeypatch.setattr(hardware, "CGROUP_V2_CPU_MAX", path)


def _v1(tmp_path, monkeypatch, quota: int, period: int):
    quota_path, period_path = tmp_path / "cfs_quota_us", tmp_path / "cfs_period_us"
    quota_path.write_text(str(quota))
    period_path.write_text(str(period))
    monkeypatch.setattr(hardware, "CGROUP_V1_CPU_QUOTA", quota_path)
    monkeypatch.setattr(hardware, "CGROUP_V1_CPU_PERIOD", period_path)


def _no_cgroup_files(tmp_path, monkeypatch):
    """Point both cgroup versions at paths that don't exist, as on a host with
    neither file (or under v1 with no v2 file at all)."""
    monkeypatch.setattr(hardware, "CGROUP_V2_CPU_MAX", tmp_path / "missing-v2")
    monkeypatch.setattr(hardware, "CGROUP_V1_CPU_QUOTA", tmp_path / "missing-v1-quota")
    monkeypatch.setattr(hardware, "CGROUP_V1_CPU_PERIOD", tmp_path / "missing-v1-period")


def _affinity(monkeypatch, n: int):
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: set(range(n)))


def test_v2_quota_caps_a_wider_affinity_mask(tmp_path, monkeypatch):
    """The reported Runpod case: a 96-core affinity mask, a 12-vCPU quota."""
    _affinity(monkeypatch, 96)
    _v2(tmp_path, monkeypatch, "1200000 100000\n")  # 12 vCPUs
    assert hardware.default_thread_count() == 12


def test_v2_max_leaves_affinity_alone(tmp_path, monkeypatch):
    _affinity(monkeypatch, 8)
    _v2(tmp_path, monkeypatch, "max 100000\n")
    assert hardware.default_thread_count() == 8


def test_v2_quota_rounds_up_a_fractional_cpu(tmp_path, monkeypatch):
    _affinity(monkeypatch, 96)
    _v2(tmp_path, monkeypatch, "250000 100000\n")  # 2.5 vCPUs -> 3
    assert hardware.default_thread_count() == 3


def test_v2_quota_never_exceeds_the_affinity_mask(tmp_path, monkeypatch):
    """A quota looser than the affinity mask (e.g. a cpuset already narrower)
    is not a floor on top of it."""
    _affinity(monkeypatch, 4)
    _v2(tmp_path, monkeypatch, "6400000 100000\n")  # 64 vCPUs
    assert hardware.default_thread_count() == 4


def test_v1_quota_caps_affinity(tmp_path, monkeypatch):
    _affinity(monkeypatch, 96)
    _no_cgroup_files(tmp_path, monkeypatch)
    _v1(tmp_path, monkeypatch, quota=400000, period=100000)  # 4 vCPUs
    assert hardware.default_thread_count() == 4


def test_v1_unlimited_quota_leaves_affinity_alone(tmp_path, monkeypatch):
    _affinity(monkeypatch, 8)
    _no_cgroup_files(tmp_path, monkeypatch)
    _v1(tmp_path, monkeypatch, quota=-1, period=100000)
    assert hardware.default_thread_count() == 8


def test_missing_cgroup_files_leave_affinity_alone(tmp_path, monkeypatch):
    """Neither cgroup version's files present -- not a Linux cgroup host at
    all -- falls back to the affinity mask untouched."""
    _affinity(monkeypatch, 8)
    _no_cgroup_files(tmp_path, monkeypatch)
    assert hardware.default_thread_count() == 8


def test_v2_file_present_wins_over_v1(tmp_path, monkeypatch):
    """v2 is checked first; a v1-only quota file is irrelevant once a v2 file
    parses, even one with a looser (or absent) cap."""
    _affinity(monkeypatch, 96)
    _v2(tmp_path, monkeypatch, "max 100000\n")
    _v1(tmp_path, monkeypatch, quota=400000, period=100000)
    assert hardware.default_thread_count() == 96


def test_quota_never_drops_below_one(tmp_path, monkeypatch):
    _affinity(monkeypatch, 8)
    _v2(tmp_path, monkeypatch, "1 100000\n")  # far less than one full CPU
    assert hardware.default_thread_count() == 1
