#!/usr/bin/env python3
"""The process every worker runs, in a remote container or as a local
subprocess of the dashboard.

In a container, the image's bootstrap (docker-setup/worker/bootstrap.py)
starts this after unpacking the bundle; for a local slot, the dashboard starts
it directly. It handles the process concerns (environment, results sink,
SIGTERM, the params/provenance record) and then hands off to the role's runner
from the workload registry (scribblez/workloads/).

SIGTERM (docker stop, a spot interruption, a slot paused in the dashboard)
raises WorkerStopped out of the runner's loop. Runners flush completed output
and exit, losing at most the cycle in flight.

The results sink (SCZ_SINK, cloud/sinks.py) decides where output goes: "r2"
uploads it to the bucket, "local" moves it into the tag's tree on this
machine's mount dir.

All configuration comes from environment variables:

    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,      bucket credentials (r2 sink only)
    R2_SECRET_ACCESS_KEY, R2_BUCKET
    SCZ_WORKLOAD                          workload name (default "kill_test")
    SCZ_ROLE                              role name (default: the workload's first)
    SCZ_TAG                               run tag (required)
    SCZ_SINK                              "r2" (default) or "local"
    SCZ_<PARAM>                           workload params (scribblez/params.py
                                          encoding; defaults from the dataclass)
    SCZ_THREADS                           worker threads (default: all cores)
    SCZ_MAX_CYCLES                        stop after N cycles (default 0 = run
                                          until stopped)
    SCZ_WORKER_ID                         manifest/stats identity (default: the
                                          hostname)
    SCZ_WORKER_KIND                       slot kind reported in stats: "local"
                                          or "ssh" (default: the sink's)
    SCZ_BUNDLE                            bundle reference for the bootstrap
                                          ("latest" or a bundle_id); unused here
    SCZ_BUNDLE_ID, SCZ_HOST_ARCH,         set by the bootstrap; recorded in the
    SCZ_BUNDLE_ARCH                       params record and stats
    SCZ_DEVICE                            torch device for a train role
                                          (default "cuda"; read by the trainers)
    SCZ_MOUNT_ROOT                        root of the tag trees (default: the
                                          mount dir). Lets an r2-sink trainer
                                          run on the controller's own machine
                                          without writing into the tag tree
                                          the controller manages.
"""

import os
import signal
import socket
import sys
from dataclasses import asdict
from pathlib import Path

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.hardware import default_thread_count
from scribblez.workloads.worker import WorkerStopped

from cloud.sinks import make_sink

# The SCZ_* variables that configure the worker rather than the workload
# (documented in the module docstring). Every other SCZ_* variable must be a
# parameter this bundle's schema knows.
WORKER_ENV_VARS = (
    "SCZ_WORKLOAD",
    "SCZ_ROLE",
    "SCZ_TAG",
    "SCZ_SINK",
    "SCZ_THREADS",
    "SCZ_MAX_CYCLES",
    "SCZ_WORKER_ID",
    "SCZ_WORKER_KIND",
    "SCZ_BUNDLE",
    "SCZ_BUNDLE_ID",
    "SCZ_HOST_ARCH",
    "SCZ_BUNDLE_ARCH",
    "SCZ_DEVICE",
    "SCZ_MOUNT_ROOT",
)


# The exit code after a SIGTERM, whatever the runner returned while draining.
# The dashboard reads exit 0 as "the role reached its end condition" (the slot
# is finished and not restarted), so an interrupted worker must not return it.
# 143 is the conventional code for death by SIGTERM (128 + 15).
EXIT_INTERRUPTED = 143
_interrupted = False


def _on_sigterm(signum, frame):
    global _interrupted
    _interrupted = True
    raise WorkerStopped


def check_params_understood(spec, env):
    """Refuse to start when the environment carries workload parameters this
    bundle's schema does not know.

    The controller builds the environment from its own schema, so an unknown
    parameter means the bundle is older than the controller. from_env would
    ignore it, and the worker would quietly produce data under settings
    different from the rest of the fleet. Failing at startup turns a stale
    deployment into a clear error rather than a corpus to discard later.
    """
    unknown = params_mod.unknown_env(spec.params_cls, env, allowed=WORKER_ENV_VARS)
    assert not unknown, (
        f"bundle {env.get('SCZ_BUNDLE_ID', '(unknown)')} does not understand "
        f"{', '.join(unknown)}: it predates parameters the controller is sending. "
        "Push a bundle built from the controller's code and recreate this worker."
    )


def worker_id() -> str:
    return os.environ.get("SCZ_WORKER_ID") or socket.gethostname()


def provenance() -> dict:
    return {
        "bundle_id": os.environ.get("SCZ_BUNDLE_ID"),
        "host_arch": os.environ.get("SCZ_HOST_ARCH"),
        "bundle_arch": os.environ.get("SCZ_BUNDLE_ARCH"),
    }


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    spec = workloads.get(os.environ.get("SCZ_WORKLOAD", "kill_test"))
    role = spec.role(os.environ.get("SCZ_ROLE", spec.roles[0].name))
    check_params_understood(spec, os.environ)
    try:
        tag = os.environ["SCZ_TAG"]
        params = params_mod.from_env(spec.params_cls)
        threads = int(os.environ.get("SCZ_THREADS", 0)) or default_thread_count()
        mount_root = Path(os.environ["SCZ_MOUNT_ROOT"]) if "SCZ_MOUNT_ROOT" in os.environ else None
        sink = make_sink(spec, tag, mount_root)
        kind = os.environ.get("SCZ_WORKER_KIND") or sink.kind
        if role.deps:
            workloads.resolve(role.deps)(params)
        wid = worker_id()
        sink.push_json(
            f"params/{wid}.json",
            {
                "worker_id": wid,
                "workload": spec.name,
                "role": role.name,
                "tag": tag,
                "params": asdict(params),
                "threads": threads,
                "kind": kind,
                **provenance(),
            },
        )
        ctx = workloads.WorkerContext(
            spec=spec,
            role=role,
            tag=tag,
            params=params,
            worker_id=wid,
            kind=kind,
            threads=threads,
            max_cycles=int(os.environ.get("SCZ_MAX_CYCLES", 0)),
            sink=sink,
            provenance=provenance(),
            mount_root=mount_root,
        )
        code = workloads.resolve(role.runner)(ctx)
    except WorkerStopped:
        print("SIGTERM during startup; exiting")
        return EXIT_INTERRUPTED
    return EXIT_INTERRUPTED if _interrupted else code


if __name__ == "__main__":
    sys.exit(main())
