#!/usr/bin/env python3
"""Probe exported position_eval ONNX checkpoints for FP16 headroom.

The standalone face of the export gate (docs/fp16_safe_serving.md): the same
probe the trainer runs at every export, applied to already-exported files.
Run it across a run's checkpoint series to see whether the peaks plateau or
double per thousand epochs, or on a single suspect file (the known-bad
face-up-official ep4414 fails it; a healthy checkpoint passes with room).

Each model is probed under its own metadata-declared input arm, over the
committed GCG evaluation set swept across extreme current-score leads --
where magnitudes peak. Exits non-zero if any probed file exceeds the gate
threshold.

Usage:
    ./py/scripts/position_eval/fp16_probe.py model_epoch_*.onnx
    ./py/scripts/position_eval/fp16_probe.py suspect.onnx --top 12
"""

import argparse
import sys

import onnx
from scribblez.ffi import (
    InputArm,
    session_input_arm,
    set_contingent_features,
    set_opp_leave_input,
)
from scribblez.fp16_gate import FP16_MAX, GATE_THRESHOLD, intermediate_peaks
from scribblez.position_eval import analysis
from scribblez.position_eval.onnx_export import fp16_probe_feeds
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def model_arm(path: str) -> InputArm:
    """The arm an exported model consumes, from its ONNX metadata_props and
    declared input widths (the dashboard's _position_eval_model_arm, without
    holding an inference session open per file)."""
    m = onnx.load(path, load_external_data=False)
    meta = {p.key: p.value for p in m.metadata_props}
    dims = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    return InputArm(
        contingent_features=meta.get("contingent_features") == "true",
        opp_leave_input=meta.get("opp_leave_input") == "true",
        spatial_planes=dims["input_spatial"][1],
        scalar_size=dims["input_scalar"][1],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("models", nargs="+", help="Exported position_eval ONNX file(s).")
    p.add_argument("--dataset", default=str(analysis.DEFAULT_DATASET), help="GCG dataset dir.")
    p.add_argument("--threshold", type=float, default=GATE_THRESHOLD, help="Gate threshold.")
    p.add_argument("--top", type=int, default=6, help="Worst tensors listed per model.")
    args = p.parse_args()

    # One arm per invocation: the probe rows are encoded under the (first)
    # model's own arm, adopted into the session so the score-diff scalar
    # offset the sweep stamps is that arm's.
    arm = model_arm(args.models[0])
    set_contingent_features(arm.contingent_features)
    set_opp_leave_input(arm.opp_leave_input)
    session_arm = session_input_arm()
    if (session_arm.spatial_planes, session_arm.scalar_size) != (
        arm.spatial_planes,
        arm.scalar_size,
    ):
        sys.exit(f"error: {args.models[0]} declares widths the session arm does not encode")

    names, inputs = analysis.load_inputs(args.dataset, session_arm)
    if not names:
        sys.exit(f"error: no GCG positions in {args.dataset}")
    feeds = fp16_probe_feeds(inputs, session_arm.spatial_planes)
    print(f"probe: {len(names)} positions from {args.dataset}, lead-swept\n")

    failed = False
    for path in args.models:
        if model_arm(path) != arm:
            sys.exit(f"error: {path} declares a different input arm than {args.models[0]}")
        peaks = intermediate_peaks(path, feeds)
        peak = peaks[0][1] if peaks else 0.0
        verdict = "FAIL" if peak > args.threshold else "pass"
        if peak > args.threshold:
            failed = True
        print(
            f"{path}: peak {peak:.0f} [{verdict}] "
            f"(threshold {args.threshold:.0f}, fp16 max {FP16_MAX:.0f})"
        )
        for name, value in peaks[: args.top]:
            print(f"    {value:10.0f}  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
