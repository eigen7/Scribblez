"""The container environment for a worker that runs from a bundle.

The dashboard (scribblez/dashboard/workers.py) passes this to every ssh-slot
container it creates. The worker entrypoint reads it (see its docstring for
the list).
"""

from scribblez import workloads


def bundle_worker_env(
    spec: workloads.WorkloadSpec,
    tag: str,
    params,
    *,
    role: str,
    bundle_id: str,
    worker_id: str,
) -> dict[str, str]:
    """The workload's SCZ_* definition, the bundle the container is given
    (recorded in its stats; the caller adds SCZ_BUNDLE_ARCH, the build it
    copies in), and the slot's identity. The kind is set explicitly because
    the worker cannot infer it from its sink."""
    return {
        **spec.worker_env(tag, params, role),
        "SCZ_BUNDLE_ID": bundle_id,
        "SCZ_WORKER_ID": worker_id,
        "SCZ_WORKER_KIND": "ssh",
    }
