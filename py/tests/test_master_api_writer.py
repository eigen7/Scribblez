"""The control plane's handlers under the writer rule, as the dashboard runs
them: a manager whose blocking thread is the one writer (control_store.py). Every
change must reach the stores as a command, and every read, served on the event
loop, must change nothing; either mistake trips the writer check, which a
handler answers with a 400."""

import json
import shutil
import tempfile
import threading
from pathlib import Path

import tornado.testing
from cloud.bundles import BundleManifest
from scribblez.dashboard import api
from scribblez.dashboard.tag_queue import TagQueue
from scribblez.dashboard.workers import WorkerManager

_TAG = {"workload": "position_eval", "tag": "t"}


class WriterRuleTest(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.mount_root = Path(tempfile.mkdtemp())
        self.manager = WorkerManager(self.mount_root)
        self.manager.claim_writer()
        self.queue = TagQueue(self.manager)
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.queue.shutdown()
        self.manager.shutdown()
        shutil.rmtree(self.mount_root, ignore_errors=True)

    def get_app(self):
        return api.make_app(str(self.mount_root), self.manager, self.queue)

    def post(self, path: str, body: dict) -> dict:
        response = self.fetch(path, method="POST", body=json.dumps(body), raise_error=False)
        assert response.code == 200, response.body
        return json.loads(response.body)

    def get(self, path: str) -> dict:
        response = self.fetch(path, raise_error=False)
        assert response.code == 200, response.body
        return json.loads(response.body)

    def test_commands_change_the_records_and_reads_see_them(self):
        self.post("/api/tasks", {**_TAG, "params": {}})
        added = self.post("/api/task/workers", {**_TAG, "kind": "local", "threads": 1})
        self.post("/api/pool/machines", {"name": "localhost"})
        task = self.get("/api/task?workload=position_eval&tag=t")
        assert [w["worker_id"] for w in task["workers"]] == added["added"]
        pool = self.get("/api/pool")
        assert [m["name"] for m in pool["machines"]] == ["localhost"]
        self.get("/api/queue")
        self.get("/api/queue/plan?workload=position_eval&tag=t")
        self.get("/api/cloud/stop_all")
        self.get("/api/workload_tags?workload=position_eval")

    def test_redeploy_pins_the_live_task(self):
        """Redeploy loads the task on the writer: pinning a reader's copy
        would save it over the live record."""
        self.manager._build_bundle = lambda archs: BundleManifest(
            bundle_id="b-1", git_sha="0", git_dirty=False, archs=archs, source_hash="h"
        )
        self.post("/api/tasks", {**_TAG, "params": {}})
        assert self.post("/api/task/deploy", _TAG) == {"bundle_id": "b-1"}
        assert self.get("/api/task?workload=position_eval&tag=t")["bundle_id"] == "b-1"

    def test_a_read_is_answered_while_a_command_holds_the_writer(self):
        """A read waits on nothing the writer is doing: the Machine pool page
        answers while a long step (an upload, a build) runs."""
        release = threading.Event()
        self.manager._blocking.submit(release.wait)
        try:
            self.get("/api/pool")
            self.get("/api/queue")
        finally:
            release.set()
