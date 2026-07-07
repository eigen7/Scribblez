"""Machine hardware introspection shared by scripts that size worker pools."""

import os


def default_thread_count() -> int:
    """Number of logical processors available to this process (its CPU affinity
    mask, so taskset/cgroup cpusets are respected).

    This is the project-wide default for worker-thread counts: compute-bound
    pools (self-play game generation, Monte Carlo workers) default to using
    every available logical processor. The C++ counterpart is
    util::default_thread_count() (util/hardware.h).
    """
    return len(os.sched_getaffinity(0))
