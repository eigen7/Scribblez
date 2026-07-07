"""The WorkerManager: reconciles worker slots with real processes and pods.

Owned by the dashboard API process. Local worker slots are backed by
subprocesses of this process running the worker entrypoint with the local
results sink; cloud slots are backed by Runpod pods (one pod per slot),
created through the same pod spec as the fleet CLI. While a task has any cloud
slots, a cloud_sync --watch subprocess streams that tag's bucket results into
the local mount.

Desired state lives in task.json (dashboard/tasks.py); actual state is
observed live (subprocess poll, pod GET). reconcile() closes the gap: it
relaunches local workers that should be running (e.g. after a dashboard
restart) and restarts interruptible pods Runpod reclaimed.

Cloud operations need <mount>/cloud/credentials.json; credentials load lazily
so a local-only dashboard works without them.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cloud.bundles import resolve_bundle_id
from cloud.credentials import CloudCredentials, load_credentials
from cloud.runpod_api import RunpodClient
from scripts.cloud_fleet import pod_create_spec

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.hardware import default_thread_count

REPO_ROOT = Path("/workspace/repo")
CLOUD_SYNC = REPO_ROOT / "py" / "scripts" / "cloud_sync.py"
SYNC_INTERVAL_SECONDS = 30


def _key(spec: workloads.WorkloadSpec, tag: str, worker_id: str = "") -> str:
    return f"{spec.name}/{tag}/{worker_id}"


def _accrue(w: tasks.WorkerRecord, running: bool, cost_per_hr: float | None):
    """Advance a slot's spend estimate to now: if it was running at its last
    observation, the elapsed interval is charged at the last observed rate.
    Every observation point (status polls, the reconcile tick, state changes)
    calls this, so the estimate only drifts across dashboard-server downtime.
    """
    now = time.time()
    if w.observed_running and w.observed_at is not None:
        w.spend += (now - w.observed_at) / 3600 * (w.cost_per_hr or 0.0)
    w.observed_at = now
    w.observed_running = running
    if cost_per_hr is not None:
        w.cost_per_hr = cost_per_hr


class WorkerManager:
    def __init__(self):
        self._local: dict[str, subprocess.Popen] = {}  # slot key -> live process
        self._sync: dict[str, subprocess.Popen] = {}  # task key -> sync watcher
        self._creds: CloudCredentials | None = None
        self._client: RunpodClient | None = None

    # ---- cloud plumbing --------------------------------------------------

    def _cloud(self) -> tuple[CloudCredentials, RunpodClient]:
        if self._client is None:
            self._creds = load_credentials()
            self._client = RunpodClient(self._creds.runpod.api_key)
        return self._creds, self._client

    def _ensure_sync(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord):
        """Keep exactly one sync watcher alive per task with cloud slots."""
        key = _key(spec, task.tag)
        has_cloud = any(w.kind == "cloud" for w in task.workers)
        proc = self._sync.get(key)
        if has_cloud and (proc is None or proc.poll() is not None):
            log = self._log_file(spec, task.tag, "cloud_sync")
            self._sync[key] = subprocess.Popen(
                [
                    sys.executable, str(CLOUD_SYNC),
                    "--workload", spec.name, "-t", task.tag,
                    "--watch", "--interval", str(SYNC_INTERVAL_SECONDS),
                ],
                stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
        elif not has_cloud and proc is not None:
            proc.terminate()
            del self._sync[key]

    # ---- local plumbing --------------------------------------------------

    def _log_file(self, spec: workloads.WorkloadSpec, tag: str, name: str):
        log_dir = spec.data_dir(tag) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return open(log_dir / f"{name}.log", "ab")

    def _spawn_local(self, spec: workloads.WorkloadSpec, task: tasks.TaskRecord, w):
        params = params_mod.validate(spec.params_cls, task.params)
        env = os.environ | spec.worker_env(task.tag, params) | {
            "SCZ_SINK": "local",
            "SCZ_THREADS": str(w.threads),
            "SCZ_WORKER_ID": w.worker_id,
        }  # fmt: skip
        log = self._log_file(spec, task.tag, w.worker_id)
        self._local[_key(spec, task.tag, w.worker_id)] = subprocess.Popen(
            [sys.executable, "-m", "cloud.worker_entrypoint"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    def _local_proc(self, spec, tag, worker_id) -> subprocess.Popen | None:
        return self._local.get(_key(spec, tag, worker_id))

    # ---- slot operations -------------------------------------------------

    def add_local(self, spec, task: tasks.TaskRecord, threads: int | None) -> tasks.WorkerRecord:
        taken = {w.worker_id for w in task.workers}
        n = next(i for i in range(len(taken) + 1) if f"local-{i}" not in taken)
        w = tasks.WorkerRecord(
            worker_id=f"local-{n}",
            kind="local",
            desired_state="running",
            threads=threads or default_thread_count(),
        )
        task.workers.append(w)
        tasks.save_task(spec, task)
        self._spawn_local(spec, task, w)
        return w

    def add_cloud(
        self, spec, task: tasks.TaskRecord, count: int, vcpus: int, flavor: str
    ) -> list[tasks.WorkerRecord]:
        creds, client = self._cloud()
        params = params_mod.validate(spec.params_cls, task.params)
        bundle_id = resolve_bundle_id(creds.r2, "latest")
        added = []
        for _ in range(count):
            name, body = pod_create_spec(
                creds, spec, task.tag, params, bundle_id=bundle_id, vcpus=vcpus, flavor=flavor
            )
            pod = client.create_pod(body)
            w = tasks.WorkerRecord(
                worker_id=name,
                kind="cloud",
                desired_state="running",
                vcpus=vcpus,
                flavor=flavor,
                pod_id=pod["id"],
            )
            task.workers.append(w)
            added.append(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)
        return added

    def set_worker_state(self, spec, task: tasks.TaskRecord, worker_id: str, run: bool):
        w = task.worker(worker_id)
        w.desired_state = "running" if run else "paused"
        _accrue(w, run, None)
        tasks.save_task(spec, task)
        if w.kind == "local":
            proc = self._local_proc(spec, task.tag, worker_id)
            if run and (proc is None or proc.poll() is not None):
                self._spawn_local(spec, task, w)
            elif not run and proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        else:
            _, client = self._cloud()
            if run:
                client.start_pod(w.pod_id)
            else:
                client.stop_pod(w.pod_id)

    def remove_worker(self, spec, task: tasks.TaskRecord, worker_id: str):
        """Remove a slot. Only non-running workers may be removed (pause
        first), so a removal never discards an in-flight cycle unannounced."""
        w = task.worker(worker_id)
        if w.kind == "local":
            proc = self._local.get(_key(spec, task.tag, worker_id))
            assert proc is None or proc.poll() is not None, (
                f"{worker_id} is running; pause it first"
            )
            self._local.pop(_key(spec, task.tag, worker_id), None)
        else:
            _, client = self._cloud()
            pod = next((p for p in client.list_pods() if p["id"] == w.pod_id), None)
            assert pod is None or pod.get("desiredStatus") != "RUNNING", (
                f"{worker_id} is running; pause it first"
            )
            if pod is not None:
                client.delete_pod(w.pod_id)
        _accrue(w, False, None)
        task.retired_spend += w.spend
        task.workers.remove(w)
        tasks.save_task(spec, task)
        self._ensure_sync(spec, task)

    # ---- observation -----------------------------------------------------

    def worker_status(self, spec, task: tasks.TaskRecord) -> list[dict]:
        """One dict per slot: the durable record plus observed live state.
        Every call is also a spend-accrual observation point (persisted)."""
        out = []
        pods = None  # fetched lazily, once, only if the task has cloud slots
        for w in task.workers:
            info = {
                "worker_id": w.worker_id,
                "kind": w.kind,
                "desired_state": w.desired_state,
                "threads": w.threads,
                "vcpus": w.vcpus,
                "flavor": w.flavor,
                "pod_id": w.pod_id,
            }
            if w.kind == "local":
                proc = self._local_proc(spec, task.tag, w.worker_id)
                running = proc is not None and proc.poll() is None
                info["state"] = "running" if running else (
                    "paused" if w.desired_state == "paused" else "exited"
                )  # fmt: skip
                _accrue(w, running, None)
            else:
                if pods is None:
                    _, client = self._cloud()
                    pods = {p["id"]: p for p in client.list_pods()}
                pod = pods.get(w.pod_id)
                if pod is None:
                    info["state"] = "terminated"
                    _accrue(w, False, None)
                else:
                    running = pod.get("desiredStatus") == "RUNNING"
                    info["state"] = "running" if running else (
                        "paused" if w.desired_state == "paused" else "interrupted"
                    )  # fmt: skip
                    info["cost_per_hr"] = pod.get("costPerHr")
                    info["public_ip"] = pod.get("publicIp")
                    info["ssh"] = f"ssh {w.pod_id}@ssh.runpod.io"
                    _accrue(w, running, pod.get("costPerHr"))
            info["spend"] = w.spend
            out.append(info)
        if task.workers:
            tasks.save_task(spec, task)
        return out

    # ---- reconciliation ----------------------------------------------------

    def _tasks_with_workers(self):
        for spec in workloads.WORKLOADS.values():
            for row in tasks.list_tags(spec):
                if not row["has_task"]:
                    continue
                task = tasks.load_task(spec, row["tag"])
                if task.workers:
                    yield spec, task

    def reconcile(self):
        """Close desired-vs-actual gaps: respawn local workers that should be
        running (dashboard restart, crashed process) and restart interruptible
        pods Runpod reclaimed. Runs at boot and periodically; doubling as a
        spend-accrual heartbeat (worker_status persists it) even when no
        browser is polling."""
        for spec, task in self._tasks_with_workers():
            status = self.worker_status(spec, task)
            for w, info in zip(task.workers, status, strict=True):
                if w.desired_state != "running":
                    continue
                if w.kind == "local" and info["state"] == "exited":
                    self._spawn_local(spec, task, w)
                elif w.kind == "cloud" and info["state"] == "interrupted":
                    _, client = self._cloud()
                    client.start_pod(w.pod_id)
            self._ensure_sync(spec, task)

    def shutdown(self):
        """Stop owned subprocesses (workers flush completed pairs on SIGTERM);
        pods are unaffected -- cloud work continues across dashboard restarts."""
        for proc in [*self._local.values(), *self._sync.values()]:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
