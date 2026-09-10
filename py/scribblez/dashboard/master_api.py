"""Control-plane handlers for the master dashboard.

The read-only training data plane lives in api.py; these handlers add the
master flow: enumerating workloads (with their param schemas and role
declarations, which drive the web forms), creating tasks, managing worker
slots through the process-wide WorkerManager (settings["worker_manager"]), and
the generic per-role worker Stats data. Registered alongside the data plane by
api.make_app().

Expected client errors (bad params, unknown tags, missing cloud credentials,
Runpod failures) return 400 with {"error": ...} rather than a stack trace.
"""

import json
import time

import tornado.web
from bokeh.embed import json_item
from cloud.credentials import CredentialsError
from cloud.runpod_api import RunpodError, fetch_cloud_offers
from cloud.ssh_machine import SshMachineError
from scripts.cloud_fleet import CpuResources, GpuResources

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks, worker_stats_figures

# Exception types that describe a bad request or unavailable dependency, not a
# server bug; their message is the response.
_CLIENT_ERRORS = (
    AssertionError,
    KeyError,
    params_mod.ParamsError,
    CredentialsError,
    RunpodError,
    SshMachineError,
)


def _role_payload(role: workloads.RoleSpec) -> dict:
    return {
        "name": role.name,
        "title": role.title,
        "singleton": role.singleton,
        "kinds": list(role.kinds),
        "interruptible": role.interruptible,
        "gpu": role.gpu,
        "stats": ({"unit": role.stats.unit, "phases": role.stats.phases} if role.stats else None),
    }


def _cloud_resources(body: dict) -> CpuResources | GpuResources:
    """The pod hardware selection from an add-worker request body: a GPU type +
    count when the form posted a gpu_type_id, otherwise a CPU flavor + vCPUs."""
    if body.get("gpu_type_id"):
        gpu_count = int(body.get("gpu_count", 1))
        return GpuResources(gpu_type_id=body["gpu_type_id"], gpu_count=gpu_count)
    return CpuResources(vcpus=int(body.get("vcpus", 16)), flavor=body.get("flavor", "cpu3c"))


def _stats_by_role(spec: workloads.WorkloadSpec, tag: str) -> dict:
    """The Stats tab payload: per-role schemas plus every worker's summary."""
    records = worker_stats_figures.read_stats(spec.paths(tag).stats_dir)
    roles = {r.name: r for r in spec.roles if r.stats}
    summaries = [
        worker_stats_figures.worker_summary(rec, roles[rec["role"]].stats)
        for rec in records
        if rec.get("role") in roles
    ]
    return {
        "roles": {
            name: {"title": r.title, "unit": r.stats.unit, "phases": r.stats.phases}
            for name, r in roles.items()
        },
        "workers": summaries,
        "updated_at": max((r["updated_at"] for r in records), default=0),
    }


class _MasterBase(tornado.web.RequestHandler):
    @property
    def manager(self):
        return self.settings["worker_manager"]

    def body(self) -> dict:
        return json.loads(self.request.body or b"{}")

    def spec(self, source: dict | None = None) -> workloads.WorkloadSpec:
        name = (source or {}).get("workload") or self.get_query_argument("workload")
        return workloads.get(name)

    def guarded(self, fn):
        """Run `fn` and write its dict result; expected failures become 400s."""
        try:
            self.write(fn())
        except _CLIENT_ERRORS as e:
            self.set_status(400)
            self.write({"error": "; ".join(str(a) for a in e.args) or repr(e)})

    async def guarded_offload(self, fn):
        """`guarded`, with `fn` run off the event loop in the worker manager's
        executor. Launching and removing are seconds of ssh and cloud API
        work; the loop has to stay free to serve the status polls the operator
        is watching while they happen."""
        await self.guarded_await(self.manager.offload(fn))

    async def guarded_await(self, awaitable):
        """`guarded` for a result that is awaited rather than computed here."""
        try:
            self.write(await awaitable)
        except _CLIENT_ERRORS as e:
            self.set_status(400)
            self.write({"error": "; ".join(str(a) for a in e.args) or repr(e)})

    def task_or_fail(self, spec, tag: str) -> tasks.TaskRecord:
        task = tasks.load_task(spec, tag)
        assert task is not None, f"tag '{tag}' has no task record"
        return task


class WorkloadsHandler(_MasterBase):
    def get(self):
        self.write(
            {
                "workloads": [
                    {
                        "name": spec.name,
                        "title": spec.title,
                        "params": params_mod.public_schema(spec.params_cls),
                        "primary_params": list(spec.primary_params),
                        "profiles": spec.profiles,
                        "default_profile": spec.default_profile,
                        "roles": [_role_payload(r) for r in spec.roles],
                    }
                    for spec in workloads.WORKLOADS.values()
                ]
            }
        )


class WorkloadTagsHandler(_MasterBase):
    def get(self):
        self.guarded(lambda: {"tags": tasks.list_tags(self.spec())})


class TaskCreateHandler(_MasterBase):
    def post(self):
        body = self.body()

        def create():
            task = tasks.create_task(
                self.spec(body), body.get("tag", ""), body.get("params", {}), body.get("profile")
            )
            return {"tag": task.tag}

        self.guarded(create)


class TaskHandler(_MasterBase):
    def get(self):
        spec = self.spec()
        tag = self.get_query_argument("tag")

        def info():
            task = tasks.load_task(spec, tag)
            workers = self.manager.worker_status(spec, task) if task else []
            spend = task.retired_spend + sum(w.spend for w in task.workers) if task else 0.0
            return {
                "workload": spec.name,
                "tag": tag,
                "has_task": task is not None,
                "params": task.params if task else None,
                "profile": task.profile if task else "",
                "profile_diff": spec.profile_diff(task.profile, task.params) if task else [],
                "created_at": task.created_at if task else None,
                "progress": tasks.progress(spec, tag),
                "gates": task.gates if task else {},
                "data_dir": str(spec.data_dir(tag)),
                "workers": workers,
                "machines": self.manager.machine_status(spec, task) if task else [],
                "spend": spend,
                "bundle_id": task.bundle_id if task else None,
                "bundle_drift": self.manager.bundle_drift(task) if task else False,
            }

        self.guarded(info)


class TaskDeleteHandler(_MasterBase):
    """Delete a tag and its local data. The tag's idle worker slots go with
    it -- tearing their containers and pods down is seconds of ssh and cloud
    work, hence the offload."""

    async def post(self):
        body = self.body()

        def delete():
            self.manager.delete_task(self.spec(body), body["tag"])
            return {"ok": True}

        await self.guarded_offload(delete)


class TaskDeployHandler(_MasterBase):
    """Move a task onto the controller's current tree: build, push if the
    bucket lacks it, repin. Running remote workers are replaced with ones on
    the new bundle as reconcile next observes them."""

    async def post(self):
        body = self.body()
        spec = self.spec(body)

        async def deploy():
            task = self.task_or_fail(spec, body["tag"])
            return {"bundle_id": await self.manager.redeploy(spec, task)}

        await self.guarded_await(deploy())


class WorkerAddHandler(_MasterBase):
    def post(self):
        body = self.body()
        spec = self.spec(body)

        def add():
            task = self.task_or_fail(spec, body["tag"])
            role = body.get("role", spec.roles[0].name)
            if body.get("kind") == "local":
                added = [self.manager.add_local(spec, task, role, body.get("threads"))]
            elif body.get("kind") == "ssh":
                host = (body.get("host") or "").strip() or None
                machine = body.get("machine") or None
                assert host or machine, "ssh worker needs a host or a machine"
                added = [
                    self.manager.add_ssh(
                        spec, task, role, host=host, machine=machine, threads=body.get("threads")
                    )
                ]
            else:
                added = self.manager.add_cloud(
                    spec,
                    task,
                    role,
                    count=int(body.get("count", 1)),
                    resources=_cloud_resources(body),
                )
            return {"added": [w.worker_id for w in added]}

        self.guarded(add)


class MachineAddHandler(_MasterBase):
    """Register a machine the operator prepared, for the task's ssh slots."""

    def post(self):
        body = self.body()
        spec = self.spec(body)

        def add():
            task = self.task_or_fail(spec, body["tag"])
            m = self.manager.add_machine(
                spec,
                task,
                (body.get("name") or "").strip(),
                (body.get("host") or "").strip(),
                (body.get("identity_file") or "").strip() or None,
                int(body["gpu_count"]) if body.get("gpu_count") not in (None, "") else None,
            )
            return {"name": m.name}

        self.guarded(add)


class MachineActionHandler(_MasterBase):
    """Remove a machine and its slots -- seconds of ssh to check and clean
    each slot's container, hence the offload."""

    async def post(self):
        body = self.body()
        spec = self.spec(body)

        def act():
            task = self.task_or_fail(spec, body["tag"])
            assert body["action"] == "remove", f"unknown action '{body['action']}'"
            self.manager.remove_machine(spec, task, body["name"])
            return {"ok": True}

        await self.guarded_offload(act)


class WorkerActionHandler(_MasterBase):
    async def post(self):
        body = self.body()
        spec = self.spec(body)

        def act():
            task = self.task_or_fail(spec, body["tag"])
            action = body["action"]
            worker_ids = [body["worker_id"]] if "worker_id" in body else [
                w.worker_id for w in list(task.workers)
            ]  # fmt: skip
            for worker_id in worker_ids:
                if action == "remove":
                    self.manager.remove_worker(spec, task, worker_id)
                else:
                    assert action in ("start", "pause"), f"unknown action '{action}'"
                    self.manager.set_worker_state(spec, task, worker_id, run=action == "start")
            return {"ok": True, "workers": worker_ids}

        await self.guarded_offload(act)


class CloudOffersHandler(_MasterBase):
    """The live Runpod instance catalog (CPU flavors + GPU types with pricing
    and stock) backing the add-worker form. Cached in-process for a few minutes
    so repeatedly opening forms does not hammer the GraphQL endpoint; a fetch
    failure surfaces as a 400 the form can fall back on."""

    _CACHE_TTL = 300.0
    _cache: tuple[float, dict] | None = None

    def get(self):
        self.guarded(self._offers)

    def _offers(self) -> dict:
        cached = CloudOffersHandler._cache
        if cached is None or time.time() - cached[0] > CloudOffersHandler._CACHE_TTL:
            CloudOffersHandler._cache = (time.time(), fetch_cloud_offers())
        return CloudOffersHandler._cache[1]


class TaskStatsHandler(_MasterBase):
    def get(self):
        spec = self.spec()
        tag = self.get_query_argument("tag")
        self.guarded(lambda: _stats_by_role(spec, tag))


class TaskFigureHandler(_MasterBase):
    def get(self, name: str):
        spec = self.spec()
        tag = self.get_query_argument("tag")
        role_name = self.get_query_argument("role")

        def build():
            builder = worker_stats_figures.FIGURES.get(name)
            assert builder is not None, f"unknown figure '{name}'"
            role = spec.role(role_name)
            assert role.stats is not None, f"role '{role_name}' publishes no stats"
            records = [
                r
                for r in worker_stats_figures.read_stats(spec.paths(tag).stats_dir)
                if r.get("role") == role_name
            ]
            model = builder(records, role.stats)
            return {"item": json_item(model) if model is not None else None}

        self.guarded(build)


MASTER_ROUTES = [
    (r"/api/workloads", WorkloadsHandler),
    (r"/api/workload_tags", WorkloadTagsHandler),
    (r"/api/tasks", TaskCreateHandler),
    (r"/api/task", TaskHandler),
    (r"/api/task/delete", TaskDeleteHandler),
    (r"/api/task/deploy", TaskDeployHandler),
    (r"/api/task/workers", WorkerAddHandler),
    (r"/api/task/worker_action", WorkerActionHandler),
    (r"/api/task/machines", MachineAddHandler),
    (r"/api/task/machine_action", MachineActionHandler),
    (r"/api/cloud/offers", CloudOffersHandler),
    (r"/api/task/stats", TaskStatsHandler),
    (r"/api/task/figure/([a-z_]+)", TaskFigureHandler),
]
