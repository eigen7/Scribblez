#!/usr/bin/env python3
"""Measure how much of a teacher's footprint distributions a sparse top-k
encoding would keep.

Each placement head outputs a distribution over the 2927 footprint classes. The
.mset (and .sobs) formats could store it sparsely, as its top-k (class, value)
pairs (footprint_spatial.top_k_sparse), instead of as a dense plane. This probe
supplies the data for that choice: the .mset target planes stay dense because
the distribution is too broad for a small k (see
engine/include/training/move_set_eval_target_log.h). Rerun it when a new
teacher or mask might change that.

The distribution measured is the engine's masked footprint softmax
(ffi.masked_position_eval_placement), the exact target the student distills:
board-legality and tile-availability masks applied, illegal footprints at zero.
For each head it reports, across positions:

  * the legal support (nonzero classes after masking), a property of the mask
    rather than the model, and the k at which top-k is lossless;
  * the median and worst-case (p10) fraction of mass the top-k keeps, at each k;
  * the smallest k whose p10 clears --target.

The win heads put their not-win mass in the extra class (kExtraClass), which
counts like any other class here. For scale: a (class: u16, value: u8) sparse
entry costs 3 bytes against the dense plane's 1 byte per class, so sparse is
smaller only below ~975 entries. The sweep always extends to the largest support
seen, so full coverage is among the rows.

Usage:
    ./py/scripts/position_eval/footprint_topk_fidelity.py \
        --model /workspace/mount/tags/position_eval/<tag>/models/model_epoch_XXXX.onnx
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort
from scribblez import ffi
from scribblez import footprint_spatial as fs
from scribblez.position_eval import analysis as A

HEADS = tuple(ffi.format_layout()["constants"]["placement_head_names"])
DEFAULT_GCG_DIR = A.LARGE_DATASET


def head_distributions(sess, gcg_text, arm):
    """One position's masked footprint distributions, (len(HEADS), NUM_CLASSES):
    the teacher's raw logits put through the same masked softmax as the .mset
    target."""
    row = ffi.analyze_position_eval_gcg(gcg_text, arm)
    spatial, scalar = arm.split(row)
    sp = spatial[None].astype(np.float32)
    sc = scalar[None].astype(np.float32)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {inames[0]: sp, inames[1]: sc})
    named = dict(zip(onames, outs, strict=True))
    raw = np.stack([named[h][0] for h in HEADS])  # (H, NUM_CLASSES) raw logits
    return ffi.masked_position_eval_placement(gcg_text, raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="teacher ONNX: a position_eval export with the placement heads",
    )
    ap.add_argument(
        "--gcg-dir", default=str(DEFAULT_GCG_DIR), help="a position set (loose .gcg or part-*.gcgs)"
    )
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024],
        help="top-k widths to sweep",
    )
    ap.add_argument("--target", type=float, default=0.99, help="p10 mass fraction to clear")
    args = ap.parse_args()

    model = args.model
    gcgs = [text for _stem, text in A._dataset_items(args.gcg_dir)]
    if not gcgs:
        sys.exit(f"error: no .gcg files or part-*.gcgs bundles under {args.gcg_dir}")
    print(f"model: {model}\npositions: {len(gcgs)} from {args.gcg_dir}\n")

    ffi.set_opp_leave_input(True)
    arm = ffi.session_input_arm()
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    dists = np.stack([head_distributions(sess, text, arm) for text in gcgs])  # (P, H, C)

    nnz = (dists > 0).sum(axis=-1)  # (P, H)
    print("legal support (nonzero classes) per head -- the lossless sparse k:")
    for h, name in enumerate(HEADS):
        col = nnz[:, h]
        med, p90, mx = np.median(col), np.percentile(col, 90), col.max()
        print(f"  {name:>20}: median {med:5.0f}  p90 {p90:5.0f}  max {mx:5d}")
    ks = sorted(set(args.k) | {int(nnz.max())})  # always sweep up to full coverage
    print()
    print(f"top-k mass fraction (median | p10 worst-case) per head, target p10 >= {args.target}\n")
    hdr = "  k  | " + " | ".join(f"{h:>20}" for h in HEADS)
    print(hdr)
    print("-" * len(hdr))
    p10_by_k = {}
    for k in ks:
        mass = np.stack([fs.top_k_mass(dists[:, h, :], k) for h in range(len(HEADS))])  # (H, P)
        med, p10 = np.median(mass, axis=1), np.percentile(mass, 10, axis=1)
        p10_by_k[k] = p10
        cells = " | ".join(f"{med[h]:.4f} | {p10[h]:.4f}" for h in range(len(HEADS)))
        print(f" {k:>3} | {cells}")

    print()
    for h, name in enumerate(HEADS):
        ok = [k for k in ks if p10_by_k[k][h] >= args.target]
        verdict = f"k>={ok[0]}" if ok else f"none of {ks} clears p10 {args.target}"
        print(f"  {name:>20}: {verdict}")
    overall = [k for k in ks if all(p10_by_k[k][h] >= args.target for h in range(len(HEADS)))]
    best = overall[0] if overall else "NONE (raise k or lower target)"
    print(f"\n  smallest k clearing all heads: {best}")


if __name__ == "__main__":
    main()
