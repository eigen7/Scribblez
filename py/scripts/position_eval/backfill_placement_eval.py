#!/usr/bin/env python3
"""Compute a position_eval tag's placement metrics for generations that lack them.

The trainer's quality eval scores the placement heads against the Monte-Carlo
truth planes (scribblez/position_eval/analysis.placement_metrics) at every
checkpoint. Generations trained before a metric existed have only their ONNX
exports. For each such export this loads the graph's weights back into the
tag's architecture (onnx_export_util.load_onnx_initializers), runs the same
quality eval, and writes only the placement metrics into the tag's
dashboard.db, so the dashboard's Loss tab covers the whole run.

Generations that already have every placement metric are skipped unless
--force. The first rebuilt model is checked against onnxruntime on a few
positions: an architecture-param mismatch can survive the weight loading when
the tensor names happen to match, and would otherwise produce plausible but
wrong numbers.

Usage:
    ./py/scripts/position_eval/backfill_placement_eval.py -t mytag
"""

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from scribblez import workloads
from scribblez.dashboard import db
from scribblez.ffi import get_input_shapes, set_opp_leave_input
from scribblez.onnx_export_util import load_onnx_initializers
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis
from scribblez.position_eval.model import PositionEvalModel
from scribblez.position_eval.trainer import eval_position_eval_quality, load_position_eval_quality
from scribblez.spatial_trunk import transformer_config
from scribblez.train_common import timed_print

WORKLOAD = "position_eval"
# Positions the first export's rebuilt model is checked on.
PARITY_ROWS = 8
PARITY_ATOL = 1e-3


def _tag_params(paths: TagPaths):
    """The tag's frozen params from task.json, as the workload's params
    dataclass. Stored fields the dataclass lacks are dropped, so tags older
    than a param removal still load."""
    stored = json.loads((paths.root / "task.json").read_text())["params"]
    cls = workloads.get(WORKLOAD).params_cls
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in stored.items() if k in names})


def _build_model(params, device) -> tuple[PositionEvalModel, int]:
    """The tag's architecture on `device`, and the encoder's spatial plane count."""
    set_opp_leave_input(params.face_up_leaves)
    shapes = {s.name: s.dims for s in get_input_shapes()}
    model = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        num_blocks=params.num_blocks,
        trunk_channels=params.trunk_channels,
        use_film=params.use_film,
        transformer=transformer_config(dataclasses.asdict(params)),
    ).to(device)
    return model, shapes["input_spatial"][0]


def _done_epochs(conn) -> set[int]:
    """Generations whose record already carries every placement metric."""
    done = None
    for name in analysis.placement_metric_names():
        epochs = set(db.read_metric_series(conn, name)[0].tolist())
        done = epochs if done is None else done & epochs
    return done or set()


@torch.no_grad()
def _check_parity(model, onnx_path, quality: dict):
    """Check the rebuilt model against onnxruntime on the first positions. The
    torch side runs in true FP32 here: the TF32 matmuls and convolutions the
    evaluation itself uses differ from onnxruntime's CPU reference by ~1e-3,
    more than a mismatch check should forgive."""
    spatial, scalar = analysis.split_input(
        quality["inputs"][:PARITY_ROWS], quality["spatial_planes"]
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {
        "input_spatial": np.ascontiguousarray(spatial),
        "input_scalar": np.ascontiguousarray(scalar),
    }
    ort_out = session.run(None, feed)
    names = [o.name for o in session.get_outputs()]
    device = next(model.parameters()).device
    model.eval()
    matmul_precision = torch.get_float32_matmul_precision()
    conv_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch_out = model(
            torch.from_numpy(feed["input_spatial"]).to(device),
            torch.from_numpy(feed["input_scalar"]).to(device),
        )
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = conv_tf32
    for name, ref in zip(names, ort_out, strict=True):
        got = torch_out[name].float().cpu().numpy()
        if not np.allclose(got, ref, atol=PARITY_ATOL, rtol=1e-3):
            raise RuntimeError(
                f"rebuilt model disagrees with onnxruntime on '{name}' for {onnx_path} "
                f"(max |diff| {np.abs(got - ref).max():.3g}); architecture params mismatch?"
            )
    timed_print(f"parity check passed against onnxruntime on {onnx_path.name}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-t", "--tag", required=True, help="the tag whose exports to backfill")
    p.add_argument("--device", default="cuda", help="torch device for the evaluation")
    p.add_argument("--force", action="store_true", help="recompute generations already recorded")
    args = p.parse_args()

    paths = TagPaths(args.tag, WORKLOAD)
    params = _tag_params(paths)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model, spatial_planes = _build_model(params, device)
    quality = load_position_eval_quality(spatial_planes, params.face_up_leaves)
    if quality["gt"]["placement"] is None:
        timed_print("no quality dataset with placement planes: nothing to backfill")
        return 1
    conn = db.connect(paths.dashboard_db)
    skip = set() if args.force else _done_epochs(conn)
    epochs = [e for e in paths.exported_generations() if e not in skip]
    timed_print(f"{args.tag}: {len(epochs)} generations to backfill ({len(skip)} already recorded)")

    placement_names = set(analysis.placement_metric_names())
    t0 = time.time()
    for i, epoch in enumerate(epochs):
        onnx_path = paths.onnx_path(epoch)
        load_onnx_initializers(model, onnx_path)
        if i == 0:
            _check_parity(model, onnx_path, quality)
        record = eval_position_eval_quality(model, quality, device)
        db.write_metrics(conn, epoch, {k: v for k, v in record.items() if k in placement_names})
        rate = (time.time() - t0) / (i + 1)
        left_min = rate * (len(epochs) - i - 1) / 60
        timed_print(
            f"[gen {epoch}] place_l1 opp_next={record['eval_place_l1_opp_next']:.3f} "
            f"top1 opp_next={record['eval_place_top1_opp_next']:.3f}  "
            f"({i + 1}/{len(epochs)}, {rate:.1f}s/gen, ~{left_min:.0f} min left)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
