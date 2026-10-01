"""Data files a worker fetches at startup, called from the roles' `deps`
hooks (RoleSpec.deps).

Neither the worker image nor a bundle contains:

- lexica: the .kwg files encode a copyrighted wordlist and are never
  redistributed. Each machine downloads them from the public Woogles/liwords
  URL, as setup_wizard.py does for the dev machine.
- Macondo's strategy data (leave values, pre-endgame table): sparse-cloned
  from the public Macondo repo at the tag py/build.py pins.
- the eval datasets a train role scores checkpoints against: git-tracked, so
  a checkout has them; a bundle-run worker's container gets the bundle's copy
  from the dashboard, and its bootstrap unpacks it (cloud/bundles.py).

Every fetch is a no-op when its files are already present, so a restarted
worker, or a worker run inside the dev container, fetches nothing.
"""

import subprocess
import urllib.request
from pathlib import Path

from build import MACONDO_REPO_URL, MACONDO_TAG
from scribblez.paths import EVAL_POSITIONS_DIRS

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
    """Check the eval datasets (EVAL_POSITIONS_DIRS) are under the repo root:
    a checkout's own copy for a local slot or a CLI, and for a bundle-run
    worker the bundle's copy, which the dashboard copies into the container of
    each role that reads them (WorkerManager._run_ssh_container)."""
    missing = [d for d in EVAL_POSITIONS_DIRS if not d.is_dir()]
    assert not missing, f"eval datasets missing: {missing}"
