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
    """CPU count implied by the cgroup CPU quota, rounded up, or None when no
    quota is in force or no cgroup CPU files exist."""
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
    """The project-wide default size for compute-bound thread pools: the
    logical processors in this process's affinity mask, capped by the cgroup
    CPU quota when one is set.

    The quota cap matters in containers limited by quota rather than cpuset,
    such as rented cloud machines: a 12-vCPU container on a 96-core host
    reports an affinity mask of 96, and a pool sized from that oversubscribes
    it 8x. The C++ counterpart is util::default_thread_count() (util/misc.h).
    """
    affinity = len(os.sched_getaffinity(0))
    quota_cpus = _cgroup_quota_cpus()
    if quota_cpus is None:
        return affinity
    return max(1, min(affinity, quota_cpus))
