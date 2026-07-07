#!/usr/bin/env python3
"""Long-running entrypoint for a cloud worker.

Launched by the worker image's bootstrap (docker-setup/worker/bootstrap.py)
after it unpacks a code+binary bundle at /workspace/repo. Fetches runtime data
dependencies, records a provenance manifest in the results bucket, then loops:
run one generation cycle, upload each completed output pair, delete the local
copy. SIGTERM (pod stop / preemption) stops the loop, finishes uploading
already-completed pairs, and exits cleanly -- at most the in-flight cycle is
lost, matching the local generator's interruption semantics.

Configuration is entirely via environment variables:

    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,     results-bucket credentials
    R2_SECRET_ACCESS_KEY, R2_BUCKET
    SCZ_WORKLOAD                          workload name (default "kill_test")
    SCZ_TAG                               run tag (required)
    SCZ_THREADS                           worker threads (default: all cores)
    SCZ_GAMES_PER_BATCH, SCZ_ROLLOUTS,    kill-test knobs (defaults:
    SCZ_TOP_K, SCZ_POSITIONS_PER_GAME,    KillTestParams field defaults;
    SCZ_OPEN_LEAVES                       OPEN_LEAVES: "1" to enable)
    SCZ_MAX_CYCLES                        stop after N cycles (default 0 = run
                                          until stopped)
    SCZ_WORKER_ID                         manifest/log identity (default: the
                                          Runpod pod id, else hostname)
    SCZ_BUNDLE_ID, SCZ_HOST_ARCH,         set by the bootstrap; recorded in
    SCZ_BUNDLE_ARCH                       the manifest
"""

import json
import os
import signal
import socket
import sys
from dataclasses import asdict
from pathlib import Path

from scribblez.hardware import default_thread_count
from scripts.generate_kill_test_data import KillTestParams, run_one_cycle, slog_dir

from cloud import worker_deps
from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone


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


def upload_manifest(r2: R2Credentials, tag: str, params: KillTestParams):
    """Record this worker's exact configuration and code provenance under the
    tag, making the tag self-describing and cross-worker consistency checkable."""
    manifest = {
        "worker_id": worker_id(),
        "workload": os.environ.get("SCZ_WORKLOAD", "kill_test"),
        "tag": tag,
        "params": asdict(params),
        "bundle_id": os.environ.get("SCZ_BUNDLE_ID"),
        "host_arch": os.environ.get("SCZ_HOST_ARCH"),
        "bundle_arch": os.environ.get("SCZ_BUNDLE_ARCH"),
    }
    res = rclone(
        r2,
        "rcat",
        bucket_path(r2, "kill_test", tag, "params", f"{worker_id()}.json"),
        capture=True,
        input_text=json.dumps(manifest, indent=2) + "\n",
    )
    assert res.returncode == 0, f"manifest upload failed: {res.stderr}"


def upload_completed_pairs(r2: R2Credentials, out_dir: Path, tag: str) -> int:
    """Upload every complete .slog/.sobs pair in `out_dir` to the tag's bucket
    prefix and delete the local copies, returning the number of pairs moved.

    The .sobs uploads before its .slog: a .slog missing its sidecar reads as
    pending work downstream (a local generator would re-sim it; kill_test.py
    skips it), while an orphaned .sobs is inert -- so the bucket, like the
    local directory, only ever presents complete pairs plus inert leftovers.
    """
    moved = 0
    for sobs in sorted(out_dir.glob("*.sobs")):
        slog = sobs.with_suffix(".slog")
        for f in (sobs, slog):
            res = rclone(
                r2,
                "copyto",
                str(f),
                bucket_path(r2, "kill_test", tag, "slogs", f.name),
                capture=True,
            )
            assert res.returncode == 0, f"upload of {f.name} failed: {res.stderr}"
        slog.unlink()
        sobs.unlink()
        moved += 1
    return moved


def params_from_env() -> KillTestParams:
    def get_int(var: str, default: int) -> int:
        return int(os.environ.get(var, default))

    return KillTestParams(
        threads=get_int("SCZ_THREADS", default_thread_count()),
        games_per_batch=get_int("SCZ_GAMES_PER_BATCH", KillTestParams.games_per_batch),
        rollouts=get_int("SCZ_ROLLOUTS", KillTestParams.rollouts),
        top_k=get_int("SCZ_TOP_K", KillTestParams.top_k),
        positions_per_game=get_int("SCZ_POSITIONS_PER_GAME", KillTestParams.positions_per_game),
        open_leaves=os.environ.get("SCZ_OPEN_LEAVES") == "1",
    )


def run_kill_test(r2: R2Credentials) -> int:
    tag = os.environ["SCZ_TAG"]
    params = params_from_env()
    max_cycles = int(os.environ.get("SCZ_MAX_CYCLES", 0))

    worker_deps.fetch_kill_test_deps()
    upload_manifest(r2, tag, params)

    out_dir = slog_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"worker {worker_id()}: generating tag '{tag}' with {params}")

    cycle = 0
    try:
        # A restarted worker may find completed pairs a previous run generated
        # but never uploaded; move them out first.
        upload_completed_pairs(r2, out_dir, tag)
        while max_cycles == 0 or cycle < max_cycles:
            cycle += 1
            rc = run_one_cycle(out_dir, params)
            if rc != 0:
                return rc
            moved = upload_completed_pairs(r2, out_dir, tag)
            print(f"cycle {cycle}: uploaded {moved} pair(s)")
    except WorkerStopped:
        moved = upload_completed_pairs(r2, out_dir, tag)
        print(f"SIGTERM: uploaded {moved} completed pair(s); exiting")
    return 0


WORKLOADS = {
    "kill_test": run_kill_test,
}


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    workload = os.environ.get("SCZ_WORKLOAD", "kill_test")
    r2 = r2_from_env()
    try:
        return WORKLOADS[workload](r2)
    except WorkerStopped:
        print("SIGTERM during startup; exiting")
        return 0


if __name__ == "__main__":
    sys.exit(main())
