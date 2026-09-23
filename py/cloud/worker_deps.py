"""Data files a worker fetches at startup, called from the roles' `deps`
hooks (RoleSpec.deps).

Neither the worker image nor a bundle contains:

- lexica: the .kwg files encode a copyrighted wordlist and are never
  redistributed. Each machine downloads them from the public Woogles/liwords
  URL, as setup_wizard.py does for the dev machine.
- Macondo's strategy data (leave values, pre-endgame table): sparse-cloned
  from the public Macondo repo at the tag py/build.py pins.
- the eval datasets a train role scores checkpoints against: git-tracked, so
  a checkout has them, but a bundle-run worker takes them from the bucket's
  deps/ prefix (see cloud/bundles.py).

Every fetch is a no-op when its files are already present, so a restarted
worker, or a worker run inside the dev container, fetches nothing.
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from build import MACONDO_REPO_URL, MACONDO_TAG
from scribblez.paths import EVAL_POSITIONS_DIRS, REPO_ROOT

from cloud import bundles
from cloud.r2 import bucket_path, rclone
from cloud.sinks import r2_from_env

MOUNT_ROOT = Path("/workspace/mount")
LEXICA_DIR = MOUNT_ROOT / "lexica"
MACONDO_DIR = MOUNT_ROOT / "macondo"

# The engine's default lexicon (scribblez::Lexicon::Params::name). Workers run
# the engine with its defaults, so this is the one lexicon they need.
DEFAULT_LEXICON = "NWL23"

# Duplicates setup_common.LIWORDS_KWG_URL_TEMPLATE: setup_common lives at the
# repo root, outside the py/ tree that bundles ship.
LIWORDS_KWG_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/woogles-io/liwords/master/"
    "liwords-ui/public/wasm/2024/{name}.kwg"
)


def fetch_lexicon(name: str):
    """Download <mount>/lexica/<name>.kwg. It lands under a temporary name
    and is renamed into place, so a partial download is never visible."""
    kwg = LEXICA_DIR / f"{name}.kwg"
    if kwg.is_file():
        return
    LEXICA_DIR.mkdir(parents=True, exist_ok=True)
    url = LIWORDS_KWG_URL_TEMPLATE.format(name=name)
    print(f"fetching lexicon {name} from {url}")
    tmp = kwg.with_suffix(".kwg.partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(kwg)


def fetch_macondo_strategy(lexicon: str):
    """Ensure <mount>/macondo has the leave values for `lexicon` and the
    shared pre-endgame table.

    The engine reads these files straight from a Macondo checkout
    (hasty_equity.cpp) and needs no Macondo build, so a sparse, blobless,
    depth-1 clone of just those two directories suffices."""
    strategy = MACONDO_DIR / "data" / "strategy"
    leaves = strategy / lexicon / "leaves.klv2"
    peg = strategy / "default" / "preendgame.json"
    if leaves.is_file() and peg.is_file():
        return
    if not MACONDO_DIR.is_dir():
        print(f"sparse-cloning Macondo {MACONDO_TAG} into {MACONDO_DIR}")
        subprocess.check_call(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--branch",
                MACONDO_TAG,
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                MACONDO_REPO_URL,
                str(MACONDO_DIR),
            ]
        )
    subprocess.check_call(
        [
            "git",
            "-C",
            str(MACONDO_DIR),
            "sparse-checkout",
            "add",
            f"data/strategy/{lexicon}",
            "data/strategy/default",
        ]
    )
    assert leaves.is_file() and peg.is_file(), (
        f"Macondo strategy data for {lexicon} missing after sparse checkout"
    )


def fetch_eval_positions():
    """Ensure the eval datasets (EVAL_POSITIONS_DIRS) under the repo root are
    the version this worker's bundle was deployed with.

    Without a bundle (a local slot, a CLI) the checkout's own copy is used and
    must exist. A bundle-run worker (SCZ_BUNDLE_ID set) compares its copy's
    digest with the manifest's and replaces the copy from the bucket when they
    differ, as they do after a redeploy that changed the datasets."""
    bundle_id = os.environ.get("SCZ_BUNDLE_ID")
    if not bundle_id:
        missing = [d for d in EVAL_POSITIONS_DIRS if not d.is_dir()]
        assert not missing, f"eval datasets missing from the checkout: {missing}"
        return
    r2 = r2_from_env()
    manifest = bundles.read_manifest(r2, bundle_id)
    assert manifest is not None and manifest.eval_positions, (
        f"bundle {bundle_id} names no eval datasets; it predates them. Redeploy."
    )
    want = manifest.eval_positions
    if all(d.is_dir() for d in EVAL_POSITIONS_DIRS) and bundles.eval_positions_digest() == want:
        return
    print(f"fetching eval datasets {want} from the bucket")
    with tempfile.TemporaryDirectory(prefix="scribblez-positions-") as tmp:
        tar_path = Path(tmp) / "positions.tar.gz"
        res = rclone(
            r2, "copyto", bucket_path(r2, bundles.eval_positions_object(want)), str(tar_path)
        )
        assert res.returncode == 0, f"download of the eval datasets {want} failed"
        for d in EVAL_POSITIONS_DIRS:
            shutil.rmtree(d, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(REPO_ROOT, filter="data")
    got = bundles.eval_positions_digest()
    assert got == want, f"eval datasets unpacked at {got}, manifest names {want}"


def fetch_kill_test_deps():
    """Everything a kill-test worker needs beyond the bundle itself."""
    fetch_lexicon(DEFAULT_LEXICON)
    fetch_macondo_strategy(DEFAULT_LEXICON)
