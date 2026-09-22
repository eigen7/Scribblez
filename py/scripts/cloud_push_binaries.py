#!/usr/bin/env python3
"""Upload the locally built engine binaries + py/ tree to the results bucket.

Packages the engine binaries workers run (cloud.bundles.BUNDLE_BINARY_NAMES) and the
py/ tree into a git-SHA-stamped bundle under bundles/<bundle_id>/ in R2, and
points bundles/LATEST at it. Remote worker containers download the bundle at startup,
so this -- not a Docker push -- is the code-deployment step: build with
py/build.py, push with this; the dashboard pins tasks to what it deploys.

The bundle contains exactly what was last built: uncommitted local changes are
included (and flagged in the manifest via the -dirty bundle_id), so what runs
in the cloud is bit-for-bit what was just tested locally.

The bundle carries the archs named by --archs (default: this host's) -- each
must already be built (py/build.py --archs ...). The dashboard needs none of
this: it builds and pushes a task's bundle for the archs its machines report.

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
    parser.add_argument("--archs", default="", help="comma-separated; default: this host's")
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
