"""HTTP-level tests for the React dashboard's Tornado data API."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tornado.testing
from scribblez.dashboard import api, db
from scribblez.paths import MAX_MOVE_PER_LANE, TagPaths

# Match the shipped lane-analysis dataset size, so position indices line up.
_N_POSITIONS = 10


def _fake_lane_preds(n: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "occ": (rng.random((n, 30, 15, 27)) > 0.7).astype(np.uint8),
        "score_pmf": rng.random((n, 30, 100)).astype(np.float32),
        "has_move": rng.random((n, 30)).astype(np.float32),
    }


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
        db.write_lane_preds(
            conn, generation=0, positions=204800, preds=_fake_lane_preds(_N_POSITIONS)
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

    def test_all_figures_routable(self):
        # Every registered figure resolves (item may be null when its data isn't seeded).
        for fig in ("train_step", "throughput", "training_metrics", "positions", "calibration"):
            r = self.fetch(f"/api/figure/{fig}?task={MAX_MOVE_PER_LANE}&tag=run1")
            assert r.code == 200, fig
            assert "item" in json.loads(r.body), fig

    def test_generations_endpoint(self):
        r = self.fetch(f"/api/generations?task={MAX_MOVE_PER_LANE}&tag=run1&table=monotonicity")
        assert r.code == 200
        assert json.loads(r.body)["generations"] == []  # none seeded for this tag

    def test_generations_unknown_table_404(self):
        r = self.fetch(f"/api/generations?task={MAX_MOVE_PER_LANE}&tag=run1&table=bogus")
        assert r.code == 404

    def test_lane_positions(self):
        body = json.loads(self.fetch("/api/lane/positions").body)
        assert isinstance(body["positions"], list)  # the shipped dataset, or [] if absent

    def test_lane_generations(self):
        body = json.loads(
            self.fetch(f"/api/lane/generations?task={MAX_MOVE_PER_LANE}&tag=run1").body
        )
        assert body["generations"] == [{"generation": 0, "positions": 204800}]

    def test_lane_position_merges_truth_and_prediction(self):
        r = self.fetch(
            f"/api/lane/position?task={MAX_MOVE_PER_LANE}&tag=run1&position=0&generation=0"
        )
        if r.code == 503:
            self.skipTest("lexicon unavailable; ground truth cannot be computed")
        assert r.code == 200
        body = json.loads(r.body)
        assert {"board", "bonuses", "rack", "lanes", "on_move"} <= set(body)
        assert body["has_prediction"] is True
        assert len(body["lanes"]["rows"]) == 15 and len(body["lanes"]["cols"]) == 15
        lane = body["lanes"]["rows"][0]
        assert len(lane["pred_score_pmf"]) == 100  # prediction attached for the histogram
        assert len(lane["pred_placed"]) == 15  # per-cell predicted union for the diff

    def test_lane_position_out_of_range_404(self):
        resp = self.fetch(f"/api/lane/position?task={MAX_MOVE_PER_LANE}&tag=run1&position=999")
        assert resp.code == 404
