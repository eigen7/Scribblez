"""The leaf Positions tab's opponent panel (deep_bot_leaf_eval/opp_beliefs.py,
dashboard/deep_bot_leaf_eval_api.py's belief route): every known position
charted under the tag's fixed leave inference model on the first request,
kept in its dashboard.db, and a position the manifest gains or whose GCG
changes charted on the next."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import tornado.testing

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "forgone_play_tool").is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.dashboard import api as dashboard_api  # noqa: E402
from scribblez.dashboard import db  # noqa: E402
from scribblez.deep_bot import LEAVE, known_positions, release  # noqa: E402
from scribblez.deep_bot_leaf_eval import opp_beliefs  # noqa: E402
from scribblez.ffi import task_input_shapes  # noqa: E402
from scribblez.leave_inference.heads import HEAD_TILTED  # noqa: E402
from scribblez.leave_inference.model import LeaveInferenceModel  # noqa: E402
from scribblez.leave_inference.onnx_export import export_onnx  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402

SHAPES = {s.name: s.dims for s in task_input_shapes("leave_inference")}
POSITIONS = known_positions.load()
IDS = [p.id for p in POSITIONS]
AVERT = IDS.index("AVERT")


def export_tiny_leave_model(path: Path) -> Path:
    torch.manual_seed(0)
    net = LeaveInferenceModel(
        SHAPES["input_spatial"][0], SHAPES["input_scalar"][0], 8, 1, head=HEAD_TILTED
    ).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(
        net,
        path,
        SHAPES["input_spatial"][0],
        SHAPES["input_scalar"][0],
        leave_values="test",
        public=False,
        provenance={},
    )
    return path


@pytest.fixture(scope="module")
def leave(tmp_path_factory) -> Path:
    return export_tiny_leave_model(tmp_path_factory.mktemp("model") / "leave.onnx")


def _tag(tmp_path: Path) -> TagPaths:
    paths = TagPaths("t", "leaf_bot_generation", tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    return paths


def _held(paths: TagPaths) -> dict:
    conn = db.connect(paths.dashboard_db)
    try:
        return db.read_opp_beliefs(conn)
    finally:
        conn.close()


def _seed(paths: TagPaths, charts: dict):
    conn = db.connect(paths.dashboard_db)
    try:
        db.write_opp_beliefs(conn, charts)
    finally:
        conn.close()


def _stand_ins(but: str) -> dict:
    """Kept charts for every position but `but`: stand-ins, as their GCGs are."""
    return {p.id: (opp_beliefs.gcg_hash(p), {}) for p in POSITIONS if p.id != but}


@pytest.fixture(scope="module")
def charted(tmp_path_factory, leave) -> tuple[TagPaths, dict]:
    """A tag charted from scratch (every position, ~1 s), shared by the tests
    that read it."""
    paths = _tag(tmp_path_factory.mktemp("mount"))
    return paths, opp_beliefs.charts(paths, leave)


def test_every_position_is_charted_and_kept(charted, leave, monkeypatch):
    paths, doc = charted
    assert sorted(doc) == sorted(IDS)
    # AVERT's opponent kept four tiles; H(IN)S's bingoed, keeping none.
    assert doc["AVERT"]["leave_length"] == 4
    assert set(doc["AVERT"]["leave_value"]) == {"inferred", "uniform"}
    assert doc["H(IN)S"]["leave_value"] is None and not doc["H(IN)S"]["modelled"]
    assert {p: chart for p, (_, chart) in _held(paths).items()} == doc

    def fail(*args):
        raise AssertionError("charted again")

    monkeypatch.setattr(opp_beliefs, "chart_positions", fail)
    assert opp_beliefs.charts(paths, leave) == doc


@pytest.mark.parametrize("held", ["missing", "edited"])
def test_a_position_lacking_or_edited_is_charted_again(tmp_path, charted, leave, monkeypatch, held):
    _, doc = charted
    paths = _tag(tmp_path)
    kept = _stand_ins("AVERT")
    if held == "edited":
        kept["AVERT"] = ("an older gcg", {})
    _seed(paths, kept)
    asked = []
    chart = opp_beliefs.chart_positions

    def spy(replay, model, indices):
        asked.extend(indices)
        return chart(replay, model, indices)

    monkeypatch.setattr(opp_beliefs, "chart_positions", spy)
    assert opp_beliefs.charts(paths, leave)["AVERT"] == doc["AVERT"]
    assert asked == [AVERT]


class BeliefRouteTest(tornado.testing.AsyncHTTPTestCase):
    """A leaf_bot_generation tag's belief is release 0's leave inference
    model's."""

    def get_app(self):
        self.mount_root = self.enterContext(tempfile.TemporaryDirectory())
        paths = _tag(Path(self.mount_root))
        (paths.root / "task.json").write_text(json.dumps({"params": {}}))
        # Every other position kept already, so the request charts only AVERT.
        _seed(paths, _stand_ins("AVERT"))
        export_tiny_leave_model(release.export_path(paths, 0, LEAVE))
        return dashboard_api.make_app(self.mount_root)

    def _get(self, position: int):
        return self.fetch(
            f"/api/deep_bot_leaf_eval/belief?task=leaf_bot_generation&tag=t&position={position}"
        )

    def test_a_position_is_charted_under_release_0(self):
        r = self._get(AVERT)
        assert r.code == 200
        belief = json.loads(r.body)["belief"]
        assert belief["model"] == "release 0, the self-play's"
        assert len(belief["presence"]) == 27 and belief["leave_value"]["inferred"]["sd"] > 0

    def test_a_position_out_of_range_is_a_404(self):
        r = self._get(len(IDS))
        assert r.code == 404
        assert json.loads(r.body) == {"error": "position out of range"}
