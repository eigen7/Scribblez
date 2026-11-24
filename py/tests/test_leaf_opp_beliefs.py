"""The leaf Positions tab's opponent panel (deep_bot_leaf_eval/opp_beliefs.py,
dashboard/deep_bot_leaf_eval_api.py's belief route): every known position
charted under a generation's leave inference model on the first request,
kept on disk, and a position the manifest gains later added to the kept
charts."""

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
from scribblez.deep_bot import LEAVE, known_positions, release  # noqa: E402
from scribblez.deep_bot_leaf_eval import opp_beliefs  # noqa: E402
from scribblez.deep_bot_leaf_eval import positions as lp  # noqa: E402
from scribblez.ffi import task_input_shapes  # noqa: E402
from scribblez.leave_inference.heads import HEAD_TILTED  # noqa: E402
from scribblez.leave_inference.model import LeaveInferenceModel  # noqa: E402
from scribblez.leave_inference.onnx_export import export_onnx  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402

SHAPES = {s.name: s.dims for s in task_input_shapes("leave_inference")}
IDS = [p.id for p in known_positions.load()]


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
    return TagPaths("t", "leaf_bot_generation", tmp_path)


@pytest.fixture(scope="module")
def charted(tmp_path_factory, leave) -> tuple[TagPaths, dict]:
    """A tag's generation 20 charted from scratch (every position, ~1 s),
    shared by the tests that read it."""
    paths = _tag(tmp_path_factory.mktemp("mount"))
    return paths, opp_beliefs.charts(paths, 20, leave)


def _held_but(doc: dict, position: str) -> dict:
    """Kept charts lacking `position`: the others' entries, stand-ins."""
    return {k: v for k, v in doc.items() if k != position}


def test_every_position_is_charted_and_kept(charted, leave, monkeypatch):
    paths, doc = charted
    assert sorted(doc) == sorted(IDS)
    # AVERT's opponent kept four tiles; H(IN)S's bingoed, keeping none.
    assert doc["AVERT"]["leave_length"] == 4
    assert set(doc["AVERT"]["leave_value"]) == {"inferred", "uniform"}
    assert doc["H(IN)S"]["leave_value"] is None and not doc["H(IN)S"]["modelled"]
    assert json.loads((paths.data_dir / opp_beliefs.rel(20)).read_text()) == doc

    def fail(*args):
        raise AssertionError("charted again")

    monkeypatch.setattr(opp_beliefs, "chart_positions", fail)
    assert opp_beliefs.charts(paths, 20, leave) == doc


def test_a_position_the_kept_charts_lack_is_added(tmp_path, charted, leave, monkeypatch):
    _, doc = charted
    paths = _tag(tmp_path)
    lp.write(_held_but(doc, "AVERT"), paths.data_dir / opp_beliefs.rel(20))
    asked = []
    chart = opp_beliefs.chart_positions

    def spy(replay, model, indices):
        asked.extend(indices)
        return chart(replay, model, indices)

    monkeypatch.setattr(opp_beliefs, "chart_positions", spy)
    assert opp_beliefs.charts(paths, 20, leave) == doc
    assert asked == [IDS.index("AVERT")]


class BeliefRouteTest(tornado.testing.AsyncHTTPTestCase):
    """A leaf_bot_generation tag's belief is its release's leave inference
    model's, at an analyzed generation."""

    def get_app(self):
        self.mount_root = self.enterContext(tempfile.TemporaryDirectory())
        paths = _tag(Path(self.mount_root))
        paths.root.mkdir(parents=True)
        (paths.root / "task.json").write_text(json.dumps({"params": {}}))
        lp.write({"positions": {}}, paths.data_dir / lp.rel(20))
        # Every other position kept already, so the request charts only AVERT.
        stand_ins = _held_but(dict.fromkeys(IDS, {}), "AVERT")
        lp.write(stand_ins, paths.data_dir / opp_beliefs.rel(20))
        export_tiny_leave_model(release.export_path(paths, 20, LEAVE))
        return dashboard_api.make_app(self.mount_root)

    def _get(self, generation: int):
        return self.fetch(
            "/api/deep_bot_leaf_eval/belief?task=leaf_bot_generation&tag=t"
            f"&generation={generation}&position={IDS.index('AVERT')}"
        )

    def test_an_analyzed_generation_is_charted(self):
        r = self._get(20)
        assert r.code == 200
        belief = json.loads(r.body)["belief"]
        assert belief["model"] == "of generation 20's analysis"
        assert len(belief["presence"]) == 27 and belief["leave_value"]["inferred"]["sd"] > 0

    def test_an_unanalyzed_generation_is_a_404(self):
        r = self._get(10)
        assert r.code == 404
        assert json.loads(r.body) == {"error": "no such analysis"}
