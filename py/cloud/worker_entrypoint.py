#!/usr/bin/env python3
"""Long-running entrypoint for a worker (cloud pod or local subprocess).

On a cloud pod this is launched by the worker image's bootstrap
(docker-setup/worker/bootstrap.py) after it unpacks a code+binary bundle; the
master dashboard launches the same entrypoint as a local subprocess. It owns
the process concerns -- env parsing, the results sink, the SIGTERM handler, a
provenance manifest -- then dispatches to the runner of the requested workload
role (the workload registry, scribblez/workloads/). SIGTERM (pod stop /
preemption / dashboard pause) raises WorkerStopped out of the runner's loop;
runners flush completed output and exit cleanly, losing at most the in-flight
cycle.

The results sink (SCZ_SINK, cloud/sinks.py) decouples where output goes from
how it is made: "r2" (default) uploads to the results bucket and deletes local
copies; "local" renames output into the tag's mount-dir data tree.

Configuration is entirely via environment variables:

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
                                          Runpod pod id, else hostname)
    SCZ_BUNDLE_ID, SCZ_HOST_ARCH,         set by the cloud bootstrap; recorded
    SCZ_BUNDLE_ARCH                       in the manifest and stats
"""

import os
import signal
import socket
import sys
from dataclasses import asdict

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.hardware import default_thread_count
from scribblez.workloads.worker import WorkerStopped

from cloud.sinks import make_sink


def _on_sigterm(signum, frame):
    raise WorkerStopped


def worker_id() -> str:
    return (
        os.environ.get("SCZ_WORKER_ID") or os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
    )


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
    try:
        tag = os.environ["SCZ_TAG"]
        params = params_mod.from_env(spec.params_cls)
        threads = int(os.environ.get("SCZ_THREADS", 0)) or default_thread_count()
        sink = make_sink(spec, tag)
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
                "kind": sink.kind,
                **provenance(),
            },
        )
        ctx = workloads.WorkerContext(
            spec=spec,
            role=role,
            tag=tag,
            params=params,
            worker_id=wid,
            threads=threads,
            max_cycles=int(os.environ.get("SCZ_MAX_CYCLES", 0)),
            sink=sink,
            provenance=provenance(),
        )
        return workloads.resolve(role.runner)(ctx)
    except WorkerStopped:
        print("SIGTERM during startup; exiting")
        return 0


if __name__ == "__main__":
    sys.exit(main())
