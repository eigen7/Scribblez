#!/usr/bin/env python3
"""Build (or re-score) the large position-evaluation Monte-Carlo test set.

The dataset is a set of penultimate-bingo positions (see
docs/lexical_features_for_value.md and the position-evaluation test-set README): each
is a board where the opponent bingoed on the previous turn, so a Monte-Carlo
rollout can sample its rack cleanly from the unseen pool with no leave model. The
aggregate W/L/D and score-delta distribution over many rollouts is the
low-variance "true value" the position evaluation model should predict.

The committed artifacts live under positions/<lexicon>/<dataset>/:
  * part-NNN.gcgs  -- the harvested GCG positions, ~100 concatenated GCG blocks
                      per file (each block begins with `#character-encoding`);
  * monte-carlo-sim-results.<condition>.json -- the MC ground truth keyed by
    position stem, one file per information condition (hidden-leaves,
    face-up-leaves; see engine/include/sim/monte_carlo_sim.h). On a
    penultimate-bingo position the two coincide (the opponent kept nothing),
    so the tool writes the same truth under both names.

The loose per-position pos-*.gcg files the C++ tools consume are transient: this
script explodes the bundles into them, scores, then removes them.

Pipeline (default): harvest -> write bundles -> explode -> Monte-Carlo score.
Use --skip-harvest to re-score the committed bundles without replaying games.

Usage:
    py/scripts/build_position_eval_test_set.py                    # full build
    py/scripts/build_position_eval_test_set.py --count 1000 --games 10000
    py/scripts/build_position_eval_test_set.py --skip-harvest     # re-score bundles
"""

import argparse
import subprocess
import sys
from pathlib import Path

from util.argparse_ext import ArgumentDefaultsHelpFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_BIN = REPO_ROOT / "target" / "engine"
HARVESTER = ENGINE_BIN / "harvest_bingo_positions_tool"
MC_SIM = ENGINE_BIN / "monte_carlo_sim_tool"

# Each harvested GCG block starts with this line; it is the record boundary the
# bundles are split on. It cannot occur inside a move line or our seed-only note.
GCG_MARKER = "#character-encoding UTF-8"


def dataset_dir(lexicon: str, dataset_name: str) -> Path:
    return REPO_ROOT / "positions" / lexicon / dataset_name


def run(cmd: list[str]):
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    if rc != 0:
        sys.exit(f"command failed (exit {rc}): {' '.join(cmd)}")


def split_bundle(text: str) -> list[str]:
    """Split one bundle's text into its GCG blocks (each starting at GCG_MARKER)."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line == GCG_MARKER and current:
            blocks.append("\n".join(current).rstrip() + "\n")
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).rstrip() + "\n")
    return blocks


def explode_bundles(dataset: Path) -> int:
    """Explode every part-*.gcgs bundle into pos-NNNN.gcg files (in bundle order).
    Returns the number of positions written."""
    idx = 0
    for bundle in sorted(dataset.glob("part-*.gcgs")):
        for block in split_bundle(bundle.read_text()):
            idx += 1
            (dataset / f"pos-{idx:04d}.gcg").write_text(block)
    return idx


def clear_loose_gcgs(dataset: Path):
    for gcg in dataset.glob("pos-*.gcg"):
        gcg.unlink()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--lexicon", default="NWL23", help="Lexicon name (kwg under --lexica-dir).")
    p.add_argument(
        "--dataset-name",
        default="position-eval-test-dataset-large",
        help="Dataset dir under positions/<lexicon>/.",
    )
    p.add_argument("--count", type=int, default=1000, help="Positions to harvest.")
    p.add_argument("--per-file", type=int, default=100, help="GCG blocks per bundle file.")
    p.add_argument("--seed", type=int, default=1000000, help="Base game seed (reserved range).")
    p.add_argument("--games", type=int, default=10000, help="Monte-Carlo rollouts per position.")
    p.add_argument("--threads", type=int, default=0, help="MC worker threads (0 = tool default).")
    p.add_argument(
        "--skip-harvest",
        action="store_true",
        help="Re-score the committed bundles without replaying games.",
    )
    p.add_argument(
        "--keep-loose",
        action="store_true",
        help="Keep the exploded pos-*.gcg files instead of removing them after scoring.",
    )
    args = p.parse_args()

    if not HARVESTER.exists() or not MC_SIM.exists():
        sys.exit(f"missing built tools under {ENGINE_BIN}; run py/build.py first")

    dataset = dataset_dir(args.lexicon, args.dataset_name)
    dataset.mkdir(parents=True, exist_ok=True)

    if not args.skip_harvest:
        run(
            [
                str(HARVESTER),
                "--lexicon",
                args.lexicon,
                "--dataset-name",
                args.dataset_name,
                "--count",
                str(args.count),
                "--per-file",
                str(args.per_file),
                "--seed",
                str(args.seed),
            ]
        )

    clear_loose_gcgs(dataset)
    n = explode_bundles(dataset)
    if n == 0:
        sys.exit(f"no positions found in {dataset} (no part-*.gcgs bundles?)")
    print(f"Exploded {n} positions from bundles in {dataset}", flush=True)

    mc_cmd = [
        str(MC_SIM),
        "--lexicon",
        args.lexicon,
        "--dataset-name",
        args.dataset_name,
        "--games",
        str(args.games),
    ]
    if args.threads > 0:
        mc_cmd += ["--threads", str(args.threads)]
    run(mc_cmd)

    if not args.keep_loose:
        clear_loose_gcgs(dataset)
    print(f"Done. Ground truth: {dataset / 'monte-carlo-sim-results.<condition>.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
