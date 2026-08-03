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
        for epoch in range(1, 4):
            db.write_metrics(
                conn,
                epoch,
                {
                    "positions": 10 * epoch,
                    "loss": 1.0 / epoch,
                    "loss_score_pdf": 1.0 / epoch,
                    "score_acc": 0.5,
                },
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
        # 3 epochs x 4 metrics (positions, loss, loss_score_pdf, score_acc).
        assert json.loads(r.body)["metrics"] == 12

    def test_version_unknown_tag_404(self):
        assert self.fetch(f"/api/version?task={MAX_MOVE_PER_LANE}&tag=nope").code == 404

    def test_controls_empty_then_post_roundtrip(self):
        q = f"?task={MAX_MOVE_PER_LANE}&tag=run1"
        assert json.loads(self.fetch("/api/controls" + q).body)["controls"] == {}
        r = self.fetch(
            "/api/controls" + q,
            method="POST",
            body=json.dumps({"name": "base_lr", "value": 2.5e-4}),
        )
        assert r.code == 200
        assert json.loads(r.body)["controls"]["base_lr"] == 2.5e-4
        assert json.loads(self.fetch("/api/controls" + q).body)["controls"]["base_lr"] == 2.5e-4

    def test_controls_post_rejects_bad_body(self):
        q = f"?task={MAX_MOVE_PER_LANE}&tag=run1"
        missing_value = self.fetch(
            "/api/controls" + q, method="POST", body=json.dumps({"name": "base_lr"})
        )
        assert missing_value.code == 400
        empty_name = self.fetch(
            "/api/controls" + q, method="POST", body=json.dumps({"name": "", "value": 1.0})
        )
        assert empty_name.code == 400

    def test_figure_returns_embeddable_item(self):
        r = self.fetch(f"/api/figure/loss?task={MAX_MOVE_PER_LANE}&tag=run1")
        assert r.code == 200
        item = json.loads(r.body)["item"]
        assert {"doc", "root_id", "target_id"} <= set(item)

    def test_figure_unknown_name_404(self):
        assert self.fetch(f"/api/figure/bogus?task={MAX_MOVE_PER_LANE}&tag=run1").code == 404

    def test_all_figures_routable(self):
        # Every registered figure resolves (item may be null when its data isn't seeded).
        for fig in ("loss", "eval_quality", "training_metrics", "mset_metrics", "match_eval"):
            r = self.fetch(f"/api/figure/{fig}?task={MAX_MOVE_PER_LANE}&tag=run1")
            assert r.code == 200, fig
            assert "item" in json.loads(r.body), fig

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
