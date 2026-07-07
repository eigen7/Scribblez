#!/usr/bin/env python3
"""Long-running entrypoint for a worker (cloud pod or local subprocess).

On a cloud pod this is launched by the worker image's bootstrap
(docker-setup/worker/bootstrap.py) after it unpacks a code+binary bundle; the
master dashboard launches the same entrypoint as a local subprocess. Fetches
runtime data dependencies, records a provenance manifest, then loops: run one
generation cycle, hand completed output pairs to the results sink, and publish
a per-worker stats record. SIGTERM (pod stop / preemption / dashboard pause)
stops the loop, flushes completed pairs, and exits cleanly -- at most the
in-flight cycle is lost.

The results sink (SCZ_SINK) decouples where output goes from how it is made:

    r2 (default)  upload each pair to the results bucket and delete the local
                  copy; stats and the manifest are uploaded too
    local         pairs stay where they were generated (the mount dir IS the
                  destination); stats and the manifest are written as files

Configuration is entirely via environment variables:

    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,      bucket credentials (r2 sink only)
    R2_SECRET_ACCESS_KEY, R2_BUCKET
    SCZ_WORKLOAD                          workload name (default "kill_test")
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

import json
import os
import signal
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.hardware import default_thread_count
from scribblez.kill_test_gen import run_one_cycle

from cloud import worker_deps
from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

# Per-cycle timing samples retained in the published stats record; enough for
# recent-throughput estimates without unbounded growth.
RECENT_SAMPLES = 50


class WorkerStopped(Exception):
    """Raised out of the main loop by the SIGTERM handler."""


def _on_sigterm(signum, frame):
    raise WorkerStopped


def r2_from_env() -> R2Credentials:
    return R2Credentials(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
    )


def worker_id() -> str:
    return (
        os.environ.get("SCZ_WORKER_ID") or os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
    )


class R2Sink:
    """Uploads results to the bucket under <workload>/<tag>/ and deletes local
    copies (the bucket is the destination; the pod disk is scratch).

    Uploaded pair files carry a -<worker_id> stem suffix: the engine names
    output by per-machine nanosecond timestamp, so two workers could in
    principle mint the same name, and a bucket collision would silently splice
    one worker's .slog with another's .sobs. The suffix makes bucket (and
    therefore synced-local) names globally unique; downstream tools match
    pairs by stem, which the shared suffix preserves."""

    def __init__(self, r2: R2Credentials, workload: str, tag: str, worker_id: str):
        self._r2 = r2
        self._root = (workload, tag)
        self._worker_id = worker_id
        self.kind = "cloud"

    def push_json(self, rel_path: str, obj: dict):
        res = rclone(
            self._r2,
            "rcat",
            bucket_path(self._r2, *self._root, *rel_path.split("/")),
            capture=True,
            input_text=json.dumps(obj, indent=2) + "\n",
        )
        assert res.returncode == 0, f"upload of {rel_path} failed: {res.stderr}"

    def push_pairs(self, out_dir: Path, new_pairs: list[Path]) -> tuple[int, int, float]:
        """Upload every complete .slog/.sobs pair in `out_dir` (not just this
        cycle's -- a restarted worker flushes leftovers too) and delete local
        copies. Returns (pairs, bytes, seconds).

        The .sobs uploads before its .slog: a .slog missing its sidecar reads
        as pending work downstream, while an orphaned .sobs is inert -- so the
        bucket only ever presents complete pairs plus inert leftovers.
        """
        moved, nbytes, t0 = 0, 0, time.monotonic()
        for sobs in sorted(out_dir.glob("*.sobs")):
            slog = sobs.with_suffix(".slog")
            for f in (sobs, slog):
                nbytes += f.stat().st_size
                res = rclone(
                    self._r2,
                    "copyto",
                    str(f),
                    bucket_path(
                        self._r2, *self._root, "slogs", f"{f.stem}-{self._worker_id}{f.suffix}"
                    ),
                    capture=True,
                )
                assert res.returncode == 0, f"upload of {f.name} failed: {res.stderr}"
            slog.unlink()
            sobs.unlink()
            moved += 1
        return moved, nbytes, time.monotonic() - t0


class LocalSink:
    """Results stay where they were generated (the mount dir is the
    destination); manifests and stats are written as plain files."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self.kind = "local"

    def push_json(self, rel_path: str, obj: dict):
        path = self._data_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2) + "\n")

    def push_pairs(self, out_dir: Path, new_pairs: list[Path]) -> tuple[int, int, float]:
        return len(new_pairs), 0, 0.0


def make_sink(spec: workloads.WorkloadSpec, tag: str):
    if os.environ.get("SCZ_SINK", "r2") == "local":
        return LocalSink(spec.data_dir(tag))
    return R2Sink(r2_from_env(), spec.name, tag, worker_id())


def provenance() -> dict:
    return {
        "bundle_id": os.environ.get("SCZ_BUNDLE_ID"),
        "host_arch": os.environ.get("SCZ_HOST_ARCH"),
        "bundle_arch": os.environ.get("SCZ_BUNDLE_ARCH"),
    }


class WorkerStats:
    """The per-worker stats record published after every cycle: cumulative
    counters plus a bounded window of per-cycle timing samples. The dashboard's
    Stats tab derives throughput and bottleneck breakdowns from these."""

    def __init__(self, sink, threads: int):
        self._sink = sink
        self._record = {
            "worker_id": worker_id(),
            "kind": sink.kind,
            "threads": threads,
            "started_at": time.time(),
            "updated_at": time.time(),
            "pairs_total": 0,
            "cycles_total": 0,
            "recent": [],
            **provenance(),
        }

    def cycle_done(self, result, pairs: int, upload_bytes: int, upload_seconds: float):
        r = self._record
        r["cycles_total"] += 1
        r["pairs_total"] += pairs
        r["updated_at"] = time.time()
        r["recent"].append(
            {
                "t": r["updated_at"],
                "gen_s": round(result.gen_seconds, 3),
                "sim_s": round(result.sim_seconds, 3),
                "upload_s": round(upload_seconds, 3),
                "bytes": upload_bytes,
                "pairs": pairs,
                "pairs_total": r["pairs_total"],
            }
        )
        del r["recent"][:-RECENT_SAMPLES]
        self._sink.push_json(f"stats/{r['worker_id']}.json", r)


def run_kill_test(spec: workloads.WorkloadSpec, sink) -> int:
    tag = os.environ["SCZ_TAG"]
    params = params_mod.from_env(spec.params_cls)
    threads = int(os.environ.get("SCZ_THREADS", 0)) or default_thread_count()
    max_cycles = int(os.environ.get("SCZ_MAX_CYCLES", 0))

    worker_deps.fetch_kill_test_deps()
    sink.push_json(
        f"params/{worker_id()}.json",
        {
            "worker_id": worker_id(),
            "workload": spec.name,
            "tag": tag,
            "params": asdict(params),
            "threads": threads,
            "kind": sink.kind,
            **provenance(),
        },
    )

    out_dir = spec.data_dir(tag) / "slogs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats(sink, threads)
    print(f"worker {worker_id()} ({sink.kind}): generating tag '{tag}' with {params}")

    cycle = 0
    try:
        # A restarted worker may find completed pairs a previous run generated
        # but never delivered; flush them first.
        sink.push_pairs(out_dir, [])
        while max_cycles == 0 or cycle < max_cycles:
            cycle += 1
            result = run_one_cycle(out_dir, params, threads)
            if result.returncode != 0:
                return result.returncode
            moved, nbytes, secs = sink.push_pairs(out_dir, result.new_pairs)
            stats.cycle_done(result, moved, nbytes, secs)
            print(f"cycle {cycle}: {moved} pair(s) delivered")
    except WorkerStopped:
        moved, _, _ = sink.push_pairs(out_dir, [])
        print(f"SIGTERM: flushed {moved} completed pair(s); exiting")
    return 0


WORKLOAD_RUNNERS = {
    "kill_test": run_kill_test,
}


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    spec = workloads.get(os.environ.get("SCZ_WORKLOAD", "kill_test"))
    try:
        sink = make_sink(spec, os.environ["SCZ_TAG"])
        return WORKLOAD_RUNNERS[spec.name](spec, sink)
    except WorkerStopped:
        print("SIGTERM during startup; exiting")
        return 0


if __name__ == "__main__":
    sys.exit(main())
