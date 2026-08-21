#!/usr/bin/env python3
"""Probe whether a position-eval model binds cross-check letters to rack/leave content.

The face-up-leaves model is supposed to read a square's cross-check planes (which
letters may legally play there) *gated by* the opponent's face-up leave (which
letters the opponent actually holds). The failure mode this tool measures is the
model reading the letter mask through a fixed tile-frequency prior instead --
ignoring the leave entirely. It runs three probes against one ONNX checkpoint
(onnxruntime, CPU):

  1. LETTER SELECTIVITY -- force a hook square's cross-check mask to each single
     letter and read Pr[opp plays there]. A frequency-prior model ranks by tile
     frequency (E,A,S high; Q,Z low) regardless of the leave; a binding model
     ranks the letters the opponent holds far above the rest.
  2. AVAILABILITY SWEEP -- vary the opp-leave input for the motivating letter from
     present to absent and read the same square. A binding model swings; a
     frequency-prior model barely moves.
  3. TAIL PERCENTILES -- per-position corr(pred, MC truth) over the large set. The
     failure lives in the tail (constrained boards), so the mean is blind to it;
     split |pred - truth| by whether a cell carries cross-check bits.

The motivating case is pos-09 M7: the opponent holds G and GNU plays vertically
there (MC truth 0.668). A vertical play's constraint lives in the V (vertical)
cross-check block; the tool auto-detects which block carries a square's set.
"""

import argparse
import glob
import json

import numpy as np
import onnxruntime as ort
from scribblez import ffi
from scribblez.paths import REPO_ROOT
from scribblez.position_eval import analysis as A

# Base face-up-leaves arm (contingent_features off, opp_leave_input on): 85 planes,
# 163 scalars. Cross-check blocks and the opp-leave scalar block within that layout.
N_PLANES = 85
HCC0, VCC0, CC_END = 33, 59, 85  # horizontal / vertical cross-check plane ranges
OPP_LEAVE0 = 136  # opp-leave scalar block: OPP_LEAVE0 + (letter index 0..25)

SMALL = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset"
LARGE = REPO_ROOT / "positions" / "NWL23" / "position-eval-test-dataset-large"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def latest_checkpoint(tag: str) -> str:
    models = sorted(glob.glob(f"/workspace/mount/tags/position_eval/{tag}/models/*.onnx"))
    if not models:
        raise SystemExit(f"no ONNX checkpoints under tag {tag!r}")
    return models[-1]


def square_index(name: str) -> tuple[int, int]:
    """'M7' -> (row 6, col 12), matching the encoder's (row, col) plane layout."""
    return int(name[1:]) - 1, ord(name[0].upper()) - ord("A")


class Model:
    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inames = [i.name for i in self.sess.get_inputs()]
        self.onames = [o.name for o in self.sess.get_outputs()]

    def opp_placement(self, sp: np.ndarray, sc: np.ndarray) -> np.ndarray:
        feed = {
            self.inames[0]: sp[None].astype(np.float32),
            self.inames[1]: sc[None].astype(np.float32),
        }
        out = self.sess.run(None, feed)
        named = dict(zip(self.onames, [o[0] for o in out], strict=True))
        return sigmoid(named["opp_next_placement"])


def encode(gcg_text: str, arm) -> tuple[np.ndarray, np.ndarray]:
    row = ffi.analyze_position_eval_gcg(gcg_text, arm)
    return row[: N_PLANES * 225].reshape(N_PLANES, 15, 15).copy(), row[N_PLANES * 225 :].copy()


def hook_block(sp: np.ndarray, r: int, c: int) -> tuple[int, str]:
    """Which cross-check block carries this square's set: the non-empty one. A hook
    constrains plays along the axis perpendicular to its cross-word, so a square with
    a vertical cross-word carries its set in the V block."""
    h = int(sp[HCC0:VCC0, r, c].sum())
    v = int(sp[VCC0:CC_END, r, c].sum())
    return (VCC0, "V") if v >= h else (HCC0, "H")


def probe_letter_selectivity(model: Model, arm, square: str, focus: str):
    r, c = square_index(square)
    sp, sc = encode((SMALL / "pos-09.gcg").read_text(), arm)
    block0, block = hook_block(sp, r, c)
    print(f"\n[1] LETTER SELECTIVITY at {square} (cross-check set in {block} block)")
    base = float(model.opp_placement(sp, sc)[r, c])
    print(f"    baseline Pr[opp play] = {base:.4f}")
    scores = []
    for letter in range(26):
        x = sp.copy()
        x[HCC0:CC_END, r, c] = 0
        x[block0 + letter, r, c] = 1
        scores.append((chr(ord("A") + letter), float(model.opp_placement(x, sc)[r, c])))
    scores.sort(key=lambda t: -t[1])
    print("    ranking: " + "  ".join(f"{ltr}{p:.3f}" for ltr, p in scores))
    rank = [ltr for ltr, _ in scores].index(focus) + 1
    hi, lo = scores[0][1], scores[-1][1]
    print(f"    focus letter {focus!r} (opponent holds it): "
          f"rank {rank}/26, Pr={dict(scores)[focus]:.4f}")
    print(f"    max/min selectivity = {hi / max(lo, 1e-9):.1f} "
          f"({scores[0][0]} vs {scores[-1][0]})")
    print("    READ: a frequency prior ranks E/A/S high and the held letter low;")
    print("          binding ranks the held letter near the top.")


def probe_availability_sweep(model: Model, arm, square: str, focus: str):
    r, c = square_index(square)
    sp, sc = encode((SMALL / "pos-09.gcg").read_text(), arm)
    fi = ord(focus) - ord("A")
    print(f"\n[2] AVAILABILITY SWEEP at {square}, opp-leave letter {focus!r}")

    def m(mut) -> float:
        s = sc.copy()
        mut(s)
        return float(model.opp_placement(sp, s)[r, c])

    def clear_focus(s):
        s[OPP_LEAVE0 + fi] = 0

    def clear_block(s):
        s[OPP_LEAVE0 : OPP_LEAVE0 + 26] = 0

    print(f"    opp holds {focus} (baseline):    {m(lambda s: None):.4f}")
    print(f"    {focus} removed from opp leave:   {m(clear_focus):.4f}")
    print(f"    opp leave block zeroed:         {m(clear_block):.4f}")
    print("    READ: a binding model swings toward 0 as the letter leaves the pool.")


def probe_tail_percentiles(model: Model, arm, limit: int):
    print(f"\n[3] TAIL PERCENTILES over the large set (first {limit} positions)")
    gt = json.load(open(A.ground_truth_path(LARGE, face_up_leaves=True)))
    cors, has_bits, no_bits = [], [], []
    for stem, txt in A._dataset_items(LARGE)[:limit]:
        if stem not in gt:
            continue
        sp, sc = encode(txt, arm)
        pred = model.opp_placement(sp, sc)
        truth = np.array(gt[stem]["placement"]["opp_next_placement"]) / gt[stem]["n"]
        if truth.std() == 0 or pred.std() == 0:
            continue
        cors.append(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])
        bits = (sp[HCC0:CC_END].sum(axis=0) > 0)  # cells carrying any cross-check bit
        err = np.abs(pred - truth)
        has_bits.append(err[bits].mean() if bits.any() else np.nan)
        no_bits.append(err[~bits].mean() if (~bits).any() else np.nan)
    cors = np.array(cors)
    for q in (1, 5, 10, 25, 50):
        print(f"    p{q:<3d} corr = {np.percentile(cors, q):.3f}")
    print(f"    mean corr = {cors.mean():.3f}")
    print(f"    mean |pred-truth|  cross-check cells = {np.nanmean(has_bits):.4f}"
          f"   other cells = {np.nanmean(no_bits):.4f}")
    print("    READ: the failure is in the low percentiles and the cross-check-cell error;")
    print("          the mean is dominated by the easy majority and stays blind to it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="path to an ONNX checkpoint")
    src.add_argument("--tag", help="position_eval tag; probes its latest checkpoint")
    ap.add_argument("--square", default="M7", help="hook square to probe (default M7)")
    ap.add_argument("--focus", default="G", help="the letter the opponent holds (default G)")
    ap.add_argument("--limit", type=int, default=400, help="positions for the tail probe")
    ap.add_argument("--skip-tail", action="store_true", help="skip probe 3 (the slow one)")
    args = ap.parse_args()

    ffi.set_contingent_features(False)
    ffi.set_opp_leave_input(True)
    arm = ffi.session_input_arm()
    path = args.model or latest_checkpoint(args.tag)
    print(f"model: {path}")
    model = Model(path)
    probe_letter_selectivity(model, arm, args.square, args.focus.upper())
    probe_availability_sweep(model, arm, args.square, args.focus.upper())
    if not args.skip_tail:
        probe_tail_percentiles(model, arm, args.limit)


if __name__ == "__main__":
    main()
