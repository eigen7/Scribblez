#!/usr/bin/env python3
"""Measure, without a model, whether a transfer_test corpus carries transfer
signal: does the damage a candidate would block, read from other candidates'
probes, predict its label minus the teacher prior? See
scribblez/transfer_test/signal.py. The report is printed and written as JSON
under the tag's analyses/.

Usage: transfer_test_signal.py --tag TAG [--threads N]
"""

import argparse
import json
import os
import time

from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.transfer_test.corpus import load_corpus
from scribblez.transfer_test.signal import collect, report
from scribblez.workloads.transfer_test import CORPUS_DIR
from scribblez.workloads.transfer_test import SPEC as CORPUS_SPEC


def print_report(r: dict):
    print(
        f"{'slice':<16}{'positions':>10}{'corr':>8}{'R^2':>7}{'slope':>8}"
        f"{'share corr':>12}{'vs prior':>10}{'vs label':>10}"
    )
    for name, s in r["slices"].items():
        print(
            f"{name:<16}{s['positions']:>10}{s['corr_damage_miss']:>+8.3f}"
            f"{s['r2_damage_miss']:>7.3f}{s['slope_damage_miss']:>+8.3f}"
            f"{s['corr_share_miss']:>+12.3f}{s['corr_damage_prior']:>+10.3f}"
            f"{s['corr_damage_label']:>+10.3f}"
        )
    a = r["slices"]["all"]
    print(
        f"\ncontrol, damage blocked shuffled across candidates: corr "
        f"{r['shuffled_control_corr']:+.3f}"
    )
    print(
        f"label - prior within-row SD {a['miss_sd']:.4f} -> {a['residual_sd']:.4f} after the "
        f"damage-blocked fit; label noise SD {r['label_noise_sd']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="a transfer_test corpus tag")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    add_mount_root_argument(parser)
    args = parser.parse_args()

    paths = TagPaths(args.tag, CORPUS_SPEC.name, args.mount_root)
    files = load_corpus(paths.data_dir / CORPUS_DIR)
    t0 = time.time()

    def progress(done: int, total: int):
        print(f"\r{done}/{total} files, {time.time() - t0:.0f} s", end="", flush=True)

    candidates = collect(files, args.threads, progress)
    print()
    result = {"tag": args.tag, **report(candidates)}
    print_report(result)
    out = paths.root / "analyses" / "transfer_signal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
