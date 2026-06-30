"""HTTP-level tests for the React dashboard's Tornado data API."""

import json
import shutil
import tempfile
from pathlib import Path

import tornado.testing
from scribblez.dashboard import api, db
from scribblez.paths import MAX_MOVE_PER_LANE, TagPaths


class DashboardApiTest(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.mount_root = Path(tempfile.mkdtemp())
        paths = TagPaths("run1", MAX_MOVE_PER_LANE, mount_root=self.mount_root)
        conn = db.connect(paths.dashboard_db)
        db.write_loss_weights(conn, {"loss_score_pdf": 1.0})
        db.write_train_steps(
            conn,
            [
                {
                    "step": s,
                    "positions": 10 * s,
                    "n": 1,
                    "loss": 1.0 / s,
                    "loss_score_pdf": 1.0 / s,
                    "score_acc": 0.5,
                }
                for s in range(1, 4)
            ],
        )
        conn.close()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.mount_root, ignore_errors=True)

    def get_app(self):
        return api.make_app(str(self.mount_root))

    def test_tags_lists_the_run(self):
        body = json.loads(self.fetch(f"/api/tags?task={MAX_MOVE_PER_LANE}").body)
        assert body["tags"] == ["run1"]

    def test_version_counts_rows(self):
        r = self.fetch(f"/api/version?task={MAX_MOVE_PER_LANE}&tag=run1")
        assert r.code == 200
        assert json.loads(r.body)["train_step"] == 9  # 3 points x 3 series

    def test_version_unknown_tag_404(self):
        assert self.fetch(f"/api/version?task={MAX_MOVE_PER_LANE}&tag=nope").code == 404

    def test_figure_returns_embeddable_item(self):
        r = self.fetch(f"/api/figure/train_step?task={MAX_MOVE_PER_LANE}&tag=run1")
        assert r.code == 200
        item = json.loads(r.body)["item"]
        assert {"doc", "root_id", "target_id"} <= set(item)

    def test_figure_unknown_name_404(self):
        assert self.fetch(f"/api/figure/bogus?task={MAX_MOVE_PER_LANE}&tag=run1").code == 404
