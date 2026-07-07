#!/usr/bin/env python3
"""Upload the locally built engine binaries + py/ tree to the results bucket.

Packages target/engine/{play_game,sim_obs_tool,libscribblez_ffi.so} and the
py/ tree into a git-SHA-stamped bundle under bundles/<bundle_id>/ in R2, and
points bundles/LATEST at it. Cloud worker pods download the bundle at startup,
so this -- not a Docker push -- is the code-deployment step: build with
py/build.py, push with this, launch with scripts/cloud_fleet.py.

The bundle contains exactly what was last built: uncommitted local changes are
included (and flagged in the manifest via the -dirty bundle_id), so what runs
in the cloud is bit-for-bit what was just tested locally.

Usage:
    ./py/scripts/cloud_push_binaries.py
"""

import argparse
import sys

from cloud.bundles import push_bundle
from cloud.credentials import load_credentials
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    ).parse_args()
    creds = load_credentials()
    manifest = push_bundle(creds.r2)
    print(f"Pushed bundle {manifest.bundle_id} (now the LATEST bundle).")
    if manifest.git_dirty:
        print(
            "NOTE: the working tree has uncommitted changes; this bundle's exact "
            "source is not reproducible from git. Commit before long real runs."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
