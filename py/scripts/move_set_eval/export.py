#!/usr/bin/env python3
"""Export a move_set_eval tag's rolling checkpoint to ONNX.

The trainer exports per pass on its own; this standalone path exists for
checkpoints whose training run is already over -- a finished tag's epoch
budget is spent, so without this script its checkpoint could never produce
the ONNX the A4 runtime loads.

The model is rebuilt from the checkpoint's recorded config. Checkpoints
written since the config became self-describing carry the adopted
information-condition arm, the input widths, and the move-encoding version
directly; an older checkpoint predates those fields, so the arm is re-adopted
from the tag's .mset corpus exactly as the trainer adopted it (and the export
is stamped move_encoding_version=0 -- pre-exchange-fix rows -- which the
engine-side loader will rightly refuse to run against a newer encoder).

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
            config.update(legacy_checkpoint_condition(paths, config))
        except FileNotFoundError as e:
            sys.exit(f"error: {e}")

    model = MoveSetEvalModel(
        spatial_planes=config["spatial_planes"],
        scalar_size=config["scalar_size"],
        trunk_channels=config["trunk_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # checkpoint.save persists the post-increment cursor, so the weights on
    # disk are those of pass generation_index - 1 -- name the export the way
    # the trainer named that same pass's own export.
    last_pass = ckpt["generation_index"] - 1
    out = args.out or paths.onnx_path(last_pass)
    export_onnx(
        model,
        out,
        config["spatial_planes"],
        config["scalar_size"],
        contingent_features=config["contingent_features"],
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
