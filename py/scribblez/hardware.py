"""Machine hardware introspection shared by scripts that size worker pools."""

import math
import os
from pathlib import Path

# cgroup v2 exposes one file, "<quota> <period>" in microseconds or
# "max <period>" when unlimited.
CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")

# cgroup v1 splits the same information across two files; quota is -1 when
# unlimited.
CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cgroup_quota_cpus() -> int | None:
    """CPU count implied by the cgroup CPU quota (ceil(quota / period)), or
    None when no quota is in force -- v2's "max", v1's quota of -1, or neither
    cgroup version's files present (not running under Linux cgroups)."""
    try:
        quota_str, period_str = CGROUP_V2_CPU_MAX.read_text().split()
        if quota_str == "max":
            return None
        return math.ceil(int(quota_str) / int(period_str))
    except FileNotFoundError:
        pass

    try:
        quota = int(CGROUP_V1_CPU_QUOTA.read_text())
        period = int(CGROUP_V1_CPU_PERIOD.read_text())
    except FileNotFoundError:
        return None
    if quota < 0:
        return None
    return math.ceil(quota / period)


def default_thread_count() -> int:
    """Number of logical processors available to this process: its CPU
    affinity mask (so taskset/cgroup cpusets are respected), further capped by
    the cgroup CPU quota when one is set.

    The quota cap matters on hosts that limit CPU by cgroup quota rather than
    cpuset -- Runpod pods, notably: a 12-vCPU pod on a 96-core host reports an
    affinity mask of 96, and without the quota cap a thread pool sized from
    that oversubscribes the pod by 8x.

    This is the project-wide default for worker-thread counts: compute-bound
    pools (self-play game generation, Monte Carlo workers) default to using
    every available logical processor. The C++ counterpart is
    util::default_thread_count() (util/hardware.h).
    """
    affinity = len(os.sched_getaffinity(0))
    quota_cpus = _cgroup_quota_cpus()
    if quota_cpus is None:
        return affinity
    return max(1, min(affinity, quota_cpus))
