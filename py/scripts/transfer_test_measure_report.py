#!/usr/bin/env python3
"""Report on a transfer_test tag's step-0 measurement (docs/plans/supreme_bot_m1a.md):
the label rollout count L, the cost of a rollout, ply-one option saturation,
and coupling counts. See scribblez/transfer_test_measure.py for the method.

Usage: transfer_test_measure_report.py --tag TAG [--target 0.1] [--quantile 0.5]
"""

import argparse
from collections import Counter

import numpy as np
from scribblez import transfer_test_measure as tm
from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.workloads.transfer_test import STORE_DIR

BUDGETS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)


def budgets_for(rollouts: int) -> list[int]:
    return [n for n in BUDGETS if n < rollouts] + [rollouts]


def report_budgets(
    label: str,
    ns: list[tm.NoiseSignal],
    rollouts: int,
    candidates: int,
    target: float,
    quantile: float,
    rate: float,
):
    """Print the noise/signal table, and the rollouts a label needs with what
    a position labeled at that count costs."""
    resolvable = [s for s in ns if s.signal > 0]
    print(f"\n{label}: noise / signal of the centered labels")
    print(f"  positions with no resolvable signal: {len(ns) - len(resolvable)} of {len(ns)}")
    if not resolvable:
        return
    ratios = {
        n: np.array([tm.noise_to_signal(s, n) for s in resolvable]) for n in budgets_for(rollouts)
    }
    print(f"  {'rollouts':>9} {'median':>8} {'p75':>8} {'resolved':>9}")
    for n, r in ratios.items():
        print(
            f"  {n:>9} {np.median(r):>8.3f} {np.quantile(r, 0.75):>8.3f} "
            f"{np.mean(r <= target):>9.0%}"
        )
    # Extrapolated beyond the measured budget: noise scales as 1/n.
    needed = [s.noise_per_rollout / (target * s.signal) for s in resolvable]
    label_rollouts = float(np.quantile(needed, quantile))
    print(
        f"  rollouts for noise/signal <= {target} at the {quantile:.0%} quantile: "
        f"{label_rollouts:,.0f} (p75 {np.quantile(needed, 0.75):,.0f})"
    )
    seconds = candidates * label_rollouts / rate
    print(
        f"  labeling a {candidates}-candidate position at that count: {seconds:,.1f} s, "
        f"{3600 / seconds:,.1f} positions/hour"
    )


def report_saturation(files: list[tm.MeasuredFile]):
    print("\nPly-one options recorded per candidate board (median), by probes")
    by_stratum = tm.saturation_by_stratum(files)
    probes = sorted({n for curves in by_stratum.values() for n in curves})
    print(f"  {'stratum':>9} " + " ".join(f"{n:>6}" for n in probes))
    for stratum, curves in sorted(by_stratum.items()):
        row = " ".join(
            f"{np.median(curves[n]):>6.0f}" if n in curves else f"{'':>6}" for n in probes
        )
        print(f"  {stratum:>9} {row}")


def report_couplings(files: list[tm.MeasuredFile]):
    positions = [p for f in files for p in f.positions]
    print("\nCouplings")
    kinds = sorted({k for p in positions for k in p.offered})
    selected = Counter(c["kind"] for p in positions for c in p.couplings)
    print(
        f"  {'kind':>20} {'offered (median)':>17} {'positions offering':>19} "
        f"{'selected / position':>20}"
    )
    for k in kinds:
        offered = np.array([p.offered[k] for p in positions])
        print(
            f"  {k:>20} {np.median(offered):>17.0f} {np.mean(offered > 0):>19.0%} "
            f"{selected[k] / len(positions):>20.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the transfer_test tag")
    ap.add_argument("--target", type=float, default=0.1, help="noise / signal a label must reach")
    ap.add_argument(
        "--quantile", type=float, default=0.5, help="the share of positions that must reach it"
    )
    add_mount_root_argument(ap)
    args = ap.parse_args()

    store = TagPaths(args.tag, "transfer_test", args.mount_root).data_dir / STORE_DIR
    files = tm.read_store(store)
    positions = [p for f in files for p in f.positions]
    if not positions:
        raise SystemExit(f"no measured positions in {store}")
    rollouts = files[0].header["rollouts"]
    candidates = sum(files[0].header["recipe"][s] for s in ("top", "middle", "exchanges", "low"))
    print(f"{len(files)} files, {len(positions)} positions, {rollouts} rollouts per candidate")

    rate = tm.rollouts_per_second(files)
    print(f"\nCost: {rate:,.0f} rollouts/s on {files[0].header['threads']} threads")
    undecided = [p for p in positions if not tm.decided(p)]
    print(f"Decided positions (every rollout the same outcome): {len(positions) - len(undecided)}")
    shrink = [tm.independent_to_paired_variance(p.expected) for p in undecided]
    print(
        "Centering per rollout shrinks the expected score's per-rollout variance "
        f"{np.median(shrink):.1f}x (median)"
    )

    plausible = {"top", "middle"}
    for label, values in (("Expected score", "expected"), ("Score difference", "delta")):
        every = [tm.noise_signal(getattr(p, values)) for p in positions]
        report_budgets(
            f"{label}, every candidate",
            every,
            rollouts,
            candidates,
            args.target,
            args.quantile,
            rate,
        )
        top = [tm.noise_signal(tm.rows_in(p, getattr(p, values), plausible)) for p in positions]
        report_budgets(
            f"{label}, top and middle only",
            top,
            rollouts,
            candidates,
            args.target,
            args.quantile,
            rate,
        )
    report_saturation(files)
    report_couplings(files)


if __name__ == "__main__":
    main()
