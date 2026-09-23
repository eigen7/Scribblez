"""The placement-metric backfill script rebuilds a tag's architecture the way
the trainer does, so an export's weights load back into it."""

import importlib.util

import torch
from scribblez.paths import REPO_ROOT
from scribblez.spatial_trunk import TRUNK_TRANSFORMER
from scribblez.workloads.position_eval import PositionEvalParams

_SPEC = importlib.util.spec_from_file_location(
    "backfill_placement_eval",
    REPO_ROOT / "py" / "scripts" / "position_eval" / "backfill_placement_eval.py",
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)


def test_build_model_honors_transformer_trunk():
    # face_up_leaves=False keeps the process's FFI input layout at its default.
    params = PositionEvalParams(face_up_leaves=False, trunk=TRUNK_TRANSFORMER, num_blocks=1)
    model, _ = backfill._build_model(params, torch.device("cpu"))
    assert model.trunk.registers is not None  # only the transformer trunk has them
