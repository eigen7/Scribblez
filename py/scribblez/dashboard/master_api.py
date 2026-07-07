"""Control-plane handlers for the master dashboard.

The read-only training data plane lives in api.py; these handlers add the
master flow: enumerating workloads (with their param schemas, which drive the
web form), creating tasks, and managing worker slots through the process-wide
WorkerManager (settings["worker_manager"]). Registered alongside the data
plane by api.make_app().

Expected client errors (bad params, unknown tags, missing cloud credentials,
Runpod failures) return 400 with {"error": ...} rather than a stack trace.
"""

import json

import tornado.web
from bokeh.embed import json_item
from cloud.credentials import CredentialsError
from cloud.runpod_api import RunpodError

from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import kill_test_figures, tasks

# Exception types that describe a bad request or unavailable dependency, not a
# server bug; their message is the response.
_CLIENT_ERRORS = (
    AssertionError,
    KeyError,
    params_mod.ParamsError,
    CredentialsError,
    RunpodError,
)


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
                        "interruptible": spec.interruptible,
                        "params": params_mod.public_schema(spec.params_cls),
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
            task = tasks.create_task(self.spec(body), body.get("tag", ""), body.get("params", {}))
            return {"tag": task.tag}

        self.guarded(create)


class TaskHandler(_MasterBase):
    def get(self):
        spec = self.spec()
        tag = self.get_query_argument("tag")

        def info():
            task = tasks.load_task(spec, tag)
            slogs = spec.data_dir(tag) / "slogs"
            workers = self.manager.worker_status(spec, task) if task else []
            spend = task.retired_spend + sum(w.spend for w in task.workers) if task else 0.0
            return {
                "workload": spec.name,
                "tag": tag,
                "has_task": task is not None,
                "params": task.params if task else None,
                "created_at": task.created_at if task else None,
                "pairs": sum(1 for _ in slogs.glob("*.sobs")) if slogs.is_dir() else 0,
                "data_dir": str(spec.data_dir(tag)),
                "workers": workers,
                "spend": spend,
            }

        self.guarded(info)


class TaskDeleteHandler(_MasterBase):
    def post(self):
        body = self.body()

        def delete():
            tasks.delete_tag(self.spec(body), body["tag"])
            return {"ok": True}

        self.guarded(delete)


class WorkerAddHandler(_MasterBase):
    def post(self):
        body = self.body()
        spec = self.spec(body)

        def add():
            task = self.task_or_fail(spec, body["tag"])
            if body.get("kind") == "local":
                added = [self.manager.add_local(spec, task, body.get("threads"))]
            else:
                added = self.manager.add_cloud(
                    spec,
                    task,
                    count=int(body.get("count", 1)),
                    vcpus=int(body.get("vcpus", 16)),
                    flavor=body.get("flavor", "cpu3c"),
                )
            return {"added": [w.worker_id for w in added]}

        self.guarded(add)


class WorkerActionHandler(_MasterBase):
    def post(self):
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

        self.guarded(act)


class KillTestStatsHandler(_MasterBase):
    def get(self):
        spec = workloads.get("kill_test")
        tag = self.get_query_argument("tag")

        def stats():
            records = kill_test_figures.read_stats(spec.data_dir(tag))
            return {
                "workers": [kill_test_figures.worker_summary(r) for r in records],
                "updated_at": max((r["updated_at"] for r in records), default=0),
            }

        self.guarded(stats)


class KillTestFigureHandler(_MasterBase):
    def get(self, name: str):
        spec = workloads.get("kill_test")
        tag = self.get_query_argument("tag")

        def build():
            builder = kill_test_figures.FIGURES.get(name)
            assert builder is not None, f"unknown figure '{name}'"
            model = builder(kill_test_figures.read_stats(spec.data_dir(tag)))
            return {"item": json_item(model) if model is not None else None}

        self.guarded(build)


MASTER_ROUTES = [
    (r"/api/workloads", WorkloadsHandler),
    (r"/api/workload_tags", WorkloadTagsHandler),
    (r"/api/tasks", TaskCreateHandler),
    (r"/api/task", TaskHandler),
    (r"/api/task/delete", TaskDeleteHandler),
    (r"/api/task/workers", WorkerAddHandler),
    (r"/api/task/worker_action", WorkerActionHandler),
    (r"/api/kill_test/stats", KillTestStatsHandler),
    (r"/api/kill_test/figure/([a-z_]+)", KillTestFigureHandler),
]
