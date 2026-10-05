#!/usr/bin/env python3
"""Build the frozen teacher's prior cache for transfer_test corpus tags
(docs/plans/supreme_bot_m1a.md, PR 3): a <stem>.sprior beside each .sprobe
that lacks one. The teacher is the one each tag pinned at creation; the
torch checkpoint is checked against its ONNX export before any cache is
written. See scribblez/transfer_test/prior.py.

Usage: transfer_test_prior.py --tag TAG [--tag TAG ...] [--device cuda]
"""

import argparse
import json
import time

import torch
from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.transfer_test.prior import (
    POSITION_EVAL,
    candidate_rows,
    check_onnx_parity,
    compute_prior,
    load_teacher,
    prior_path,
    write_prior,
)
from scribblez.transfer_test.probes import read_sprobe
from scribblez.workloads.pair_store import complete_pairs
from scribblez.workloads.transfer_test import CORPUS_DIR, PROBE_EXT, SPEC

PARITY_ROWS = 64


def teacher_of(paths: TagPaths) -> tuple[str, int]:
    params = json.loads((paths.root / "task.json").read_text())["params"]
    return params["teacher_tag"], params["teacher_generation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", action="append", required=True, help="a transfer_test tag")
    parser.add_argument("--device", default="cuda")
    add_mount_root_argument(parser)
    args = parser.parse_args()

    tags = [TagPaths(tag, SPEC.name, args.mount_root) for tag in args.tag]
    teachers = {teacher_of(paths) for paths in tags}
    if len(teachers) != 1:
        raise SystemExit(f"the tags pin different teachers: {sorted(teachers)}")
    teacher_tag, generation = teachers.pop()
    device = torch.device(args.device)
    model = load_teacher(teacher_tag, generation, args.mount_root).to(device)
    onnx = TagPaths(teacher_tag, POSITION_EVAL, args.mount_root).onnx_path(generation)
    checked = False
    for paths in tags:
        files = complete_pairs(paths.data_dir / CORPUS_DIR, PROBE_EXT)
        todo = [f for f in files if not prior_path(f).exists()]
        print(f"{paths.tag}: {len(todo)} of {len(files)} files need a prior")
        start = time.time()
        for i, f in enumerate(todo):
            probes = read_sprobe(f)
            if not checked:
                check_onnx_parity(model, onnx, candidate_rows(probes)[:PARITY_ROWS])
                print(f"teacher {teacher_tag} generation {generation} matches {onnx.name}")
                checked = True
            write_prior(compute_prior(model, probes, device), prior_path(f))
            print(f"\r  {i + 1}/{len(todo)} files, {time.time() - start:.0f}s", end="", flush=True)
        if todo:
            print()


if __name__ == "__main__":
    main()
