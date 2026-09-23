#!/usr/bin/env python3
"""Does the move set student's distillation error track the cross-check change
a move causes?

The teacher (the position evaluation model) scores a candidate on its post-move
board, cross-check planes included. The student sees the pre-move board plus
the placed tiles, so it lacks the move's post-move cross-checks, above all the
hooks of the word it forms. Those could be fed in as the sparse per-move delta
of engine/include/training/cross_check_delta.h; this diagnostic measures first
whether the student's error is concentrated where that delta is large.

For each candidate it computes the student-vs-teacher error (absolute error in
win equity and score diff, WLD KL, and on a plane-carrying slice the
placement-plane KL) and two delta features:

  * changed_bits: (square, letter) cross-check bits the move flips;
  * hook_letters: letters legal after the move on the changed squares at the
    ends of the move's own word, i.e. how open that word is to a hook.

The number of tiles played drives both the features (n tiles touch up to 2n+2
squares) and the error (long plays are rarer and swingier). So each table
compares moves of one length only: they are split into terciles of the feature
and the error is compared across terciles, each cell also showing the
tercile's mean feature value. Error rising from the low to the high tercile at
a fixed length is the signal; flat rows say cross-checks are not what the
student is missing.

The default holdout slice is the one the trainer holds out
(workloads.move_set_eval.split_pairs): the full-sweep pairs when the tag has
any, which score every legal move but carry no planes. --slice train reads the
training pairs instead, which carry planes but were trained on.

Usage:
    ./py/scripts/move_set_eval/crosscheck_delta_diagnostic.py -t TAG
"""

import argparse
import sys

import torch
from scribblez import paths as paths_mod
from scribblez.move_set_eval.cross_check_diagnostic import ERRORS, FEATURES, collect, format_table
from scribblez.move_set_eval.dataset import MsetDataset, adopt_information_condition
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.paths import TagPaths
from scribblez.spatial_trunk import transformer_config
from scribblez.workloads.move_set_eval import SLOGS_DIR, split_pairs
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def load_model(paths: TagPaths, device) -> tuple[MoveSetEvalModel, dict]:
    ckpt = torch.load(paths.rolling_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = MoveSetEvalModel(
        spatial_planes=config["spatial_planes"],
        scalar_size=config["scalar_size"],
        trunk_channels=config["trunk_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        transformer=transformer_config(config),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t", "--tag", required=True, help="move_set_eval tag: its checkpoint and corpus."
    )
    p.add_argument(
        "--slice", choices=("holdout", "train"), default="holdout", help="pairs to score"
    )
    p.add_argument("--max-positions", type=int, default=20000, help="positions to score at most")
    args = p.parse_args()

    paths = TagPaths(args.tag, paths_mod.MOVE_SET_EVAL)
    device = torch.device("cuda")
    model, config = load_model(paths, device)
    train_files, holdout_files = split_pairs(paths.data_dir / SLOGS_DIR, config["holdout_every"])
    files = holdout_files if args.slice == "holdout" else train_files
    adopt_information_condition(files)
    dataset = MsetDataset(mset_files=files, with_cross_check_deltas=True)
    print(
        f"{args.tag}: {args.slice} slice, {len(files)} pairs, {dataset.num_positions} positions, "
        f"{'full-sweep' if dataset.full_sweep else 'stratified'}, planes={dataset.has_planes}"
    )

    data = collect(model, dataset, device, args.max_positions)
    print(f"scored {len(data['tiles'])} candidate moves")
    for feature in FEATURES:
        for error in ERRORS:
            if error in data:
                print("\n" + format_table(data, feature, error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
