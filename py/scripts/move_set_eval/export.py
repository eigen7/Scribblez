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
from scribblez.dataset import row_layout
from scribblez.ffi import set_contingent_features
from scribblez.move_set_eval.dataset import adopt_information_condition
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.onnx_export import export_onnx
from scribblez.move_set_eval.targets import MSET_FLAG_OPEN_LEAVES, read_mset_flags
from scribblez.paths import TagPaths
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def _legacy_condition(paths: TagPaths, config: dict) -> dict:
    """Recover the self-describing fields for a checkpoint that predates them:
    adopt the arm from the tag's .mset corpus (the trainer's own path), then
    read the input widths off the session layout. Requires the corpus."""
    mset_files = sorted(paths.data_dir.glob("slogs/*.mset"))
    if not mset_files:
        sys.exit(
            f"error: checkpoint config predates the self-describing fields and "
            f"{paths.data_dir / 'slogs'} holds no .mset to re-adopt the arm from"
        )
    set_contingent_features(config["contingent_features"])
    adopt_information_condition(mset_files)
    input_shapes, _ = row_layout()
    dims = {s.name: s.dims for s in input_shapes}
    spatial_planes = dims["input_spatial"][0]
    scalar_size = dims["input_scalar"][0]
    return {
        "open_leaves": bool(read_mset_flags(mset_files[0]) & MSET_FLAG_OPEN_LEAVES),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        # Pre-fix rows: the version constant did not exist when this trained.
        "move_encoding_version": 0,
    }


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
        config.update(_legacy_condition(paths, config))

    model = MoveSetEvalModel(
        spatial_planes=config["spatial_planes"],
        scalar_size=config["scalar_size"],
        trunk_channels=config["trunk_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # The pass index the checkpoint is at names the export, matching the
    # trainer's own per-pass naming.
    out = args.out or paths.onnx_path(ckpt["generation_index"])
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
        f"exported {ckpt_path} (pass {ckpt['generation_index']}, "
        f"{ckpt['rows_trained']} rows) -> {out}\n"
        f"  open_leaves={config['open_leaves']} "
        f"move_encoding_version={config['move_encoding_version']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
