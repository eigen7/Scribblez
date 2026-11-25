"""HTTP-level tests for the React dashboard's Tornado data API."""

import json
import shutil
import tempfile
from pathlib import Path

import tornado.testing
from bokeh.models import ColumnDataSource
from scribblez.dashboard import api, db
from scribblez.paths import POSITION_EVAL, TagPaths


class DashboardApiTest(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.mount_root = Path(tempfile.mkdtemp())
        paths = TagPaths("run1", POSITION_EVAL, mount_root=self.mount_root)
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
        conn.close()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.mount_root, ignore_errors=True)

    def get_app(self):
        return api.make_app(str(self.mount_root))

    def test_tags_lists_the_run(self):
        body = json.loads(self.fetch(f"/api/tags?task={POSITION_EVAL}").body)
        assert body["tags"] == ["run1"]

    def test_version_counts_rows(self):
        r = self.fetch(f"/api/version?task={POSITION_EVAL}&tag=run1")
        assert r.code == 200
        # 3 epochs x 4 metrics (positions, loss, loss_score_pdf, score_acc).
        assert json.loads(r.body)["metrics"] == 12

    def test_version_unknown_tag_404(self):
        assert self.fetch(f"/api/version?task={POSITION_EVAL}&tag=nope").code == 404

    def test_controls_empty_then_post_roundtrip(self):
        q = f"?task={POSITION_EVAL}&tag=run1"
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
        q = f"?task={POSITION_EVAL}&tag=run1"
        missing_value = self.fetch(
            "/api/controls" + q, method="POST", body=json.dumps({"name": "base_lr"})
        )
        assert missing_value.code == 400
        empty_name = self.fetch(
            "/api/controls" + q, method="POST", body=json.dumps({"name": "", "value": 1.0})
        )
        assert empty_name.code == 400

    def test_figure_returns_embeddable_item(self):
        r = self.fetch(f"/api/figure/loss?task={POSITION_EVAL}&tag=run1")
        assert r.code == 200
        body = json.loads(r.body)
        assert {"doc", "root_id", "target_id"} <= set(body["item"])
        assert isinstance(body["structure"], str) and body["structure"]

    def test_figure_delta_appends_then_refetches(self):
        q = f"?task={POSITION_EVAL}&tag=run1"
        structure = json.loads(self.fetch("/api/figure/loss" + q).body)["structure"]
        # Client cursors as an embed of the current figure would hold them.
        conn = db.connect(TagPaths("run1", POSITION_EVAL, self.mount_root).dashboard_db)
        model = api.FIGURES["loss"](conn, {}, str(self.mount_root))
        sources = {
            cds.name: {"n": len(cds.data["x"]), "last_x": float(list(cds.data["x"])[-1])}
            for cds in model.select({"type": ColumnDataSource})
            if cds.name
        }
        body = {"structure": structure, "sources": sources}
        db.write_metrics(conn, 4, {"positions": 40, "loss": 0.2, "loss_score_pdf": 0.2})
        conn.close()
        r = json.loads(
            self.fetch("/api/figure_delta/loss" + q, method="POST", body=json.dumps(body)).body
        )
        assert "refetch" not in r
        assert all(len(tail["x"]) == 1 for tail in r["sources"].values())
        stale = {"structure": "not-the-structure", "sources": sources}
        r = json.loads(
            self.fetch("/api/figure_delta/loss" + q, method="POST", body=json.dumps(stale)).body
        )
        assert r == {"refetch": True}

    def test_figure_delta_bad_body_400(self):
        q = f"?task={POSITION_EVAL}&tag=run1"
        assert self.fetch("/api/figure_delta/loss" + q, method="POST", body="{").code == 400

    def test_figure_unknown_name_404(self):
        assert self.fetch(f"/api/figure/bogus?task={POSITION_EVAL}&tag=run1").code == 404

    def test_all_figures_routable(self):
        # Each figure resolves; `item` is null when its data isn't seeded.
        for fig in api.FIGURES:
            r = self.fetch(f"/api/figure/{fig}?task={POSITION_EVAL}&tag=run1")
            assert r.code == 200, fig
            assert "item" in json.loads(r.body), fig

    def test_position_eval_alt_leave_without_exports_404(self):
        # "latest" resolves against the tag's exports; with none it is a 404.
        r = self.fetch(
            f"/api/position_eval/alt_leave?task={POSITION_EVAL}&tag=run1&position=0&leave=AB"
        )
        if r.code == 404 and "position out of range" in r.body.decode():
            self.skipTest("position_eval dataset unavailable")
        assert r.code == 404
        assert json.loads(r.body)["error"] == "no model generations recorded yet"
