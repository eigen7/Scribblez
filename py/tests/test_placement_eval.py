"""The placement-vs-Monte-Carlo quality metrics (position_eval/analysis.py:
placement planes ground truth, the engine collapse of a model's logits, and
placement_metrics) and the ONNX-initializer reload the backfill tool rebuilds
models through (onnx_export_util.load_onnx_initializers).

The metric arithmetic is pinned against a hand computation on synthetic
planes; the dataset side reads the committed large-set ground truth; the
collapse runs a tiny model over real dataset positions; the reload is a round
trip -- export, scramble, reload, same outputs.
"""

import numpy as np
import pytest
import torch
from scribblez.ffi import get_input_shapes, session_input_arm
from scribblez.onnx_export_util import load_onnx_initializers
from scribblez.position_eval import analysis
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES, PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx

_CPU = torch.device("cpu")
_H = len(PLACEMENT_HEAD_NAMES)


def _tiny_model() -> PositionEvalModel:
    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    return PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=2,
    ).eval()


def test_placement_metrics_match_hand_computation():
    """L1 is the legal-cell residual mass (tiles), top-1 the argmax agreement;
    cells outside the legal set are ignored on both sides, and a position whose
    MC plane is empty for a head is left out of that head."""
    n = 3
    planes = np.zeros((n, _H, 15, 15), np.float32)
    truth = np.zeros((n, _H, 15, 15), np.float32)
    legal = np.zeros((n, _H, 15, 15), bool)
    legal[:, :, 7, 5:10] = True  # a 5-cell legal strip for every head
    # position 0, head 0: model puts 0.6/0.4 on cells (7,5)/(7,6); MC 0.5/0.5
    planes[0, 0, 7, 5], planes[0, 0, 7, 6] = 0.6, 0.4
    truth[0, 0, 7, 5], truth[0, 0, 7, 6] = 0.5, 0.5
    # position 1, head 0: model's top cell differs from MC's; an illegal cell
    # carries model mass that must not count
    planes[1, 0, 7, 5], planes[1, 0, 7, 7] = 0.2, 0.3
    planes[1, 0, 0, 0] = 0.9  # illegal
    truth[1, 0, 7, 5], truth[1, 0, 7, 7] = 0.8, 0.2
    # position 2, head 0: MC plane empty -> excluded
    planes[2, 0, 7, 5] = 1.0

    m = analysis.placement_metrics(planes, truth, legal)
    l1_pos0 = 0.1 + 0.1
    l1_pos1 = 0.6 + 0.1  # the 0.9 on the illegal cell is not counted
    assert m["eval_place_l1_opp_next"] == pytest.approx((l1_pos0 + l1_pos1) / 2)
    assert m["eval_place_top1_opp_next"] == pytest.approx(0.5)  # pos 0 agrees, pos 1 does not
    # the other heads have no MC mass anywhere: recorded as nothing, not NaN
    assert set(m) == {"eval_place_l1_opp_next", "eval_place_top1_opp_next"}
    assert set(m) <= set(analysis.placement_metric_names())


def test_metric_names_stay_off_the_train_accuracy_panel():
    names = analysis.placement_metric_names()
    assert len(names) == 2 * _H
    assert not any(n.endswith("_acc") for n in names)


def test_large_set_ground_truth_carries_placement_planes():
    """The committed large set's Monte-Carlo file has the per-cell planes for
    every position, as rollout fractions, with each win plane under its plays
    plane (a won rollout is one of the rollouts)."""
    items = analysis._dataset_items(analysis.LARGE_DATASET)
    names = [stem for stem, _ in items]
    gt = analysis.load_ground_truth(analysis.LARGE_DATASET, names, face_up_leaves=True)
    planes = gt["placement"]
    assert planes is not None and planes.shape == (len(names), _H, 15, 15)
    assert planes.min() >= 0.0 and planes.max() <= 1.0
    h = {head: i for i, head in enumerate(PLACEMENT_HEAD_NAMES)}
    assert np.all(planes[:, h["opp_win_placement"]] <= planes[:, h["opp_next_placement"]] + 1e-6)
    assert np.all(planes[:, h["self_win_placement"]] <= planes[:, h["self_next_placement"]] + 1e-6)


def test_collapse_over_dataset_positions():
    """predict() carries the raw placement logits; collapsed through the engine
    they are per-cell probabilities that live only on legal cells and, for the
    plays heads, sum to the expected tile count of the move (between 0 and 7).
    Encoded under whatever arm the process-wide session already has: the
    collapse reads the board from the GCG, not the arm."""
    items = analysis._dataset_items(analysis.LARGE_DATASET)[:4]
    texts = [text for _, text in items]
    inputs = np.stack([analysis.analyze_position_eval_gcg(t, session_input_arm()) for t in texts])
    shapes = {s.name: s.dims for s in get_input_shapes()}
    model = _tiny_model()
    preds = analysis.predict(model, inputs, shapes["input_spatial"][0], _CPU)
    assert preds["placement_logits"].shape[:2] == (4, _H)
    planes = analysis.collapse_placement(preds["placement_logits"], texts)
    _, legal = analysis.load_placement_frame(analysis.LARGE_DATASET)
    legal = legal[:4]
    assert planes.shape == (4, _H, 15, 15)
    assert planes.min() >= 0.0 and planes.max() <= 1.0
    assert not np.any(planes[~legal] > 0)
    for head in ("opp_next_placement", "self_next_placement"):
        s = planes[:, PLACEMENT_HEAD_NAMES.index(head)].sum(axis=(1, 2))
        assert np.all(s > 0) and np.all(s <= 7.0 + 1e-4)


def test_load_onnx_initializers_round_trip(tmp_path):
    """Export -> scramble the weights -> reload from the graph -> outputs equal
    the original's; a graph from a different architecture is refused."""
    model = _tiny_model()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    path = tmp_path / "m.onnx"
    export_onnx(
        model, path, shapes["input_spatial"][0], shapes["input_scalar"][0], opp_leave_input=True
    )
    torch.manual_seed(1)
    spatial = torch.randn(3, shapes["input_spatial"][0], 15, 15)
    scalar = torch.randn(3, shapes["input_scalar"][0])
    with torch.no_grad():
        before = model(spatial, scalar)
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        assert not torch.allclose(model(spatial, scalar)["wld"], before["wld"])
        load_onnx_initializers(model, path)
        after = model(spatial, scalar)
    for k in before:
        assert torch.equal(after[k], before[k]), k

    other = PositionEvalModel(
        shapes["input_spatial"][0], shapes["input_scalar"][0], trunk_channels=8, num_blocks=3
    )
    with pytest.raises(ValueError, match="initializers do not match"):
        load_onnx_initializers(other, path)
