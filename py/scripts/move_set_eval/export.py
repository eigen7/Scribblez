#!/usr/bin/env python3
"""Export a move_set_eval tag's rolling checkpoint to ONNX.

The trainer exports after every pass on its own. This script is for a tag whose
training has finished: its pass budget is spent, so the trainer will never
export its checkpoint again.

The model is rebuilt from the config recorded in the checkpoint. A checkpoint
too old to record its information condition, input widths and move-encoding
version gets them recovered from the tag's corpus
(onnx_export.legacy_checkpoint_condition).

Usage:
    ./py/scripts/move_set_eval/export.py -t face-up-leaves-v1
    ./py/scripts/move_set_eval/export.py -t shakeout --out /tmp/shakeout.onnx
"""

import argparse
import sys

import torch
from scribblez import paths as paths_mod
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.onnx_export import export_onnx, legacy_checkpoint_condition
from scribblez.paths import TagPaths
from scribblez.spatial_trunk import transformer_config
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("-t", "--tag", required=True, help="move_set_eval tag to export.")
    p.add_argument("--out", default="", help="Output path (default: the tag's models/ dir).")
    args = p.parse_args()

    paths = TagPaths(args.tag, paths_mod.MOVE_SET_EVAL)
    ckpt_path = paths.rolling_checkpoint
    if not ckpt_path.exists():
        sys.exit(f"error: no rolling checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])

    if "move_encoding_version" not in config:
        try:
            config.update(legacy_checkpoint_condition(paths))
        except FileNotFoundError as e:
            sys.exit(f"error: {e}")

    model = MoveSetEvalModel(
        spatial_planes=config["spatial_planes"],
        scalar_size=config["scalar_size"],
        trunk_channels=config["trunk_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        transformer=transformer_config(config),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # The checkpoint stores the post-increment cursor, so its weights are those
    # of pass generation_index - 1; name the export as the trainer named that
    # pass's own export.
    last_pass = ckpt["generation_index"] - 1
    out = args.out or paths.onnx_path(last_pass)
    export_onnx(
        model,
        out,
        config["spatial_planes"],
        config["scalar_size"],
        opp_leave_input=config["open_leaves"],
        move_encoding_version=config["move_encoding_version"],
    )
    print(
        f"exported {ckpt_path} (pass {last_pass}, "
        f"{ckpt['rows_trained']} rows) -> {out}\n"
        f"  open_leaves={config['open_leaves']} "
        f"move_encoding_version={config['move_encoding_version']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
