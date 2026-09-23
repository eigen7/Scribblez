#!/usr/bin/env python3
"""Push the locally built engine binaries and py/ tree to the bucket by hand.

Remote workers run code from a bundle (see cloud/bundles.py): the engine
binaries they need plus the py/ tree, uploaded under bundles/<bundle_id>/ and
selected through bundles/LATEST. The dashboard builds and pushes a bundle
itself before launching a worker, for the archs its machines report, so this
script is only needed to publish a bundle without the dashboard.

The bundle holds exactly what was last built, uncommitted changes included, so
the cloud runs what was just tested locally; a dirty tree shows up as a -dirty
bundle_id. Each arch in --archs must already be built (py/build.py --archs ...).

Usage:
    ./py/scripts/cloud_push_binaries.py [--archs znver4,x86-64]
"""

import argparse
import sys

from build import detect_host_arch
from cloud.bundles import push_bundle
from cloud.credentials import load_credentials
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--archs", default="", help="comma-separated CPU archs to include (default: this host's)"
    )
    args = parser.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()] or [detect_host_arch()]
    creds = load_credentials()
    manifest = push_bundle(creds.r2, archs)
    print(f"Pushed bundle {manifest.bundle_id} (now the LATEST bundle).")
    if manifest.git_dirty:
        print(
            "NOTE: the working tree has uncommitted changes; this bundle's exact "
            "source is not reproducible from git. Commit before long real runs."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
