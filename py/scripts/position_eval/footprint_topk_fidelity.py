#!/usr/bin/env python3
"""Pick the sparse top-k width `k` for the footprint-native `.sobs`/`.mset`
formats, offline and before any format is frozen.

The footprint-native placement path stores each head's 2927-class distribution
sparsely as its top-k `(class, value)` pairs (footprint_spatial.top_k_sparse).
Truncating to top-k drops tail mass; this probe measures how much, on the teacher
softmax over a position set, so `k` is chosen from data rather than guessed.

For each head it reports, across positions, the median and the worst-case (p10)
fraction of the mass the top-k keeps, at several k -- and the smallest k whose p10
clears the target. The distribution is the engine's MASKED footprint softmax (via
ffi.masked_position_eval_placement) -- the exact target the student distills,
board-legality mask and availability applied, illegal footprints at zero -- not an
unmasked proxy. The win heads carry not-win mass in kExtraClass, reported per head.

It also reports each head's legal support -- the nonzero classes after masking,
a property of the mask rather than the model -- because that is the k at which a
sparse encoding is lossless. Against the dense plane's u8 cells, a `(class:u16,
value:u8)` entry costs 3 bytes, so sparse beats dense below ~975 entries; the
sweep always extends to the largest support seen so full coverage is on the table.

Usage:
    ./py/scripts/position_eval/footprint_topk_fidelity.py \
        --model /workspace/mount/tags/position_eval/<tag>/models/model_epoch_XXXX.onnx

The row is split into its spatial / scalar halves by the session's InputArm, so
the plane count tracks the encoder registry rather than a constant here.
"""

import argparse
import glob
import sys

import numpy as np
import onnxruntime as ort
from scribblez import ffi
from scribblez import footprint_spatial as fs
from scribblez.paths import REPO_ROOT

HEADS = tuple(ffi.format_layout()["constants"]["placement_head_names"])
DEFAULT_GCG_DIR = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset-large"


# A committed set stores its positions as part-NNN.gcgs bundles: concatenated GCG
# blocks, each opening with this pragma -- the record boundary, per the set's
# README. Loose pos-*.gcg files are a transient, git-ignored explosion of them.
GCGS_BOUNDARY = "#character-encoding UTF-8"


def position_texts(gcg_dir):
    """Every position under `gcg_dir` as GCG text: the loose .gcg files when
    present (the hand-built small set commits those), else the blocks of its
    part-*.gcgs bundles (the large set commits only those)."""
    loose = sorted(glob.glob(f"{gcg_dir}/*.gcg"))
    if loose:
        return [open(g).read() for g in loose]
    texts = []
    for part in sorted(glob.glob(f"{gcg_dir}/part-*.gcgs")):
        blocks = open(part).read().split(GCGS_BOUNDARY)
        texts += [GCGS_BOUNDARY + b for b in blocks if b.strip()]
    return texts


def head_distributions(sess, gcg_text, arm):
    """The engine's masked footprint distribution (len(HEADS), NUM_CLASSES) for
    one position: run the teacher for raw logits, then apply the same mask +
    masked-softmax the .mset target uses."""
    row = ffi.analyze_position_eval_gcg(gcg_text, arm)
    spatial, scalar = arm.split(row)  # the arm's own widths, never a hardcoded plane count
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
        "--gcg-dir", default=str(DEFAULT_GCG_DIR), help="a position set: loose .gcg or part-*.gcgs"
    )
    ap.add_argument(
        "--k", type=int, nargs="+", default=[8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024]
    )
    ap.add_argument("--target", type=float, default=0.99, help="p10 mass fraction to clear")
    args = ap.parse_args()

    model = args.model
    gcgs = position_texts(args.gcg_dir)
    if not gcgs:
        sys.exit(f"error: no .gcg files or part-*.gcgs bundles under {args.gcg_dir}")
    print(f"model: {model}\npositions: {len(gcgs)} from {args.gcg_dir}\n")

    ffi.set_opp_leave_input(True)
    arm = ffi.session_input_arm()
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    dists = np.stack([head_distributions(sess, text, arm) for text in gcgs])  # (P, H, C)

    # The mask's legal support is model-independent: it is the k at which top-k is
    # lossless, so it is the sparse size to weigh against the 2927-wide dense plane.
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
