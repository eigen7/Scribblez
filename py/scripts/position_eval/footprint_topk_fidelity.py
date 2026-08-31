#!/usr/bin/env python3
"""Pick the sparse top-k width `k` for the footprint-native `.sobs`/`.mset`
formats, offline and before any format is frozen.

The footprint-native placement path stores each head's 2927-class distribution
sparsely as its top-k `(class, value)` pairs (footprint_spatial.top_k_sparse).
Truncating to top-k drops tail mass; this probe measures how much, on the teacher
softmax over a position set, so `k` is chosen from data rather than guessed.

For each head it reports, across positions, the median and the worst-case (p10)
fraction of the softmax mass the top-k keeps, at several k -- and the smallest k
whose p10 clears the target. The softmax is unmasked (raw teacher logits): the
board-legality-masked distribution is strictly more concentrated, so unmasked
top-k mass is a conservative lower bound on what the deployed masked target keeps.
The win heads carry not-win mass in kExtraClass, so they are reported separately.

Usage:
    ./py/scripts/position_eval/footprint_topk_fidelity.py \
        --model /workspace/mount/tags/position_eval/footprints-official/models/model_epoch_XXXX.onnx
"""

import argparse
import glob
import sys

import numpy as np
import onnxruntime as ort
from scribblez import ffi
from scribblez import footprint_spatial as fs
from scribblez.paths import REPO_ROOT

N_PLANES = 85
HEADS = tuple(ffi.format_layout()["constants"]["placement_head_names"])
DEFAULT_GCG_DIR = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset"
DEFAULT_MODEL_GLOB = "/workspace/mount/tags/position_eval/footprints-official/models/*.onnx"


def softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def head_distributions(sess, gcg_text, arm):
    """Softmax distribution (len(HEADS), NUM_CLASSES) for one position."""
    row = ffi.analyze_position_eval_gcg(gcg_text, arm)
    sp = row[: N_PLANES * 225].reshape(N_PLANES, 15, 15)[None].astype(np.float32)
    sc = row[N_PLANES * 225 :][None].astype(np.float32)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {inames[0]: sp, inames[1]: sc})
    named = dict(zip(onames, outs, strict=True))
    return softmax(np.stack([named[h][0] for h in HEADS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="teacher ONNX (default: latest footprints-official)")
    ap.add_argument("--gcg-dir", default=str(DEFAULT_GCG_DIR))
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16, 32, 48, 64, 96, 128])
    ap.add_argument("--target", type=float, default=0.99, help="p10 mass fraction to clear")
    args = ap.parse_args()

    model = args.model or (sorted(glob.glob(DEFAULT_MODEL_GLOB)) or [None])[-1]
    if not model:
        sys.exit("error: no --model and no footprints-official checkpoint found")
    gcgs = sorted(glob.glob(f"{args.gcg_dir}/*.gcg"))
    if not gcgs:
        sys.exit(f"error: no .gcg positions under {args.gcg_dir}")
    print(f"model: {model}\npositions: {len(gcgs)} from {args.gcg_dir}\n")

    ffi.set_opp_leave_input(True)
    arm = ffi.session_input_arm()
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    dists = np.stack([head_distributions(sess, open(g).read(), arm) for g in gcgs])  # (P, H, C)

    print(f"top-k mass fraction (median | p10 worst-case) per head, target p10 >= {args.target}\n")
    hdr = "  k  | " + " | ".join(f"{h:>20}" for h in HEADS)
    print(hdr)
    print("-" * len(hdr))
    p10_by_k = {}
    for k in args.k:
        mass = np.stack([fs.top_k_mass(dists[:, h, :], k) for h in range(len(HEADS))])  # (H, P)
        med, p10 = np.median(mass, axis=1), np.percentile(mass, 10, axis=1)
        p10_by_k[k] = p10
        cells = " | ".join(f"{med[h]:.4f} | {p10[h]:.4f}" for h in range(len(HEADS)))
        print(f" {k:>3} | {cells}")

    print()
    for h, name in enumerate(HEADS):
        ok = [k for k in args.k if p10_by_k[k][h] >= args.target]
        verdict = f"k>={ok[0]}" if ok else f"none of {args.k} clears p10 {args.target}"
        print(f"  {name:>20}: {verdict}")
    overall = [k for k in args.k if all(p10_by_k[k][h] >= args.target for h in range(len(HEADS)))]
    best = overall[0] if overall else "NONE (raise k or lower target)"
    print(f"\n  smallest k clearing all heads: {best}")


if __name__ == "__main__":
    main()
