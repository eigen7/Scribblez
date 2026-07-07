"""Runtime data dependencies a cloud worker fetches at startup.

The worker image and bundles deliberately contain no lexica (the .kwg files
encode a copyrighted wordlist and are never redistributed; each machine
fetches them from the public Woogles/liwords URL, exactly as setup_wizard.py
does for the dev machine) and no Macondo data (fetched from the public Macondo
repo at the tag pinned in py/build.py).

Every fetch is idempotent and short-circuits when the target files already
exist -- so worker restarts skip it, and running the worker entrypoint inside
the dev container (whose mount dir setup_wizard.py/build.py already populated)
touches nothing.
"""

import subprocess
import urllib.request
from pathlib import Path

from build import MACONDO_REPO_URL, MACONDO_TAG

MOUNT_ROOT = Path("/workspace/mount")
LEXICA_DIR = MOUNT_ROOT / "lexica"
MACONDO_DIR = MOUNT_ROOT / "macondo"

# The engine's default lexicon (scribblez::Lexicon::Params.name); kill-test
# generation runs the engine with its defaults, so these are the data files a
# kill-test worker needs.
DEFAULT_LEXICON = "NWL23"

# Mirrors setup_common.LIWORDS_KWG_URL_TEMPLATE. Duplicated because
# setup_common lives at the repo root (for host-side imports) and is not part
# of worker bundles, which ship the py/ tree only.
LIWORDS_KWG_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/woogles-io/liwords/master/"
    "liwords-ui/public/wasm/2024/{name}.kwg"
)


def fetch_lexicon(name: str):
    """Download <mount>/lexica/<name>.kwg from the public liwords URL. The
    download lands under a temporary name and is renamed into place, so a
    partially-fetched file is never visible."""
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
    """Ensure Macondo's strategy data for `lexicon` (leaves) and the shared
    default/ tables (pre-endgame) exist under <mount>/macondo.

    On a fresh machine this makes a sparse, blobless, depth-1 checkout of the
    pinned Macondo tag holding just those two directories -- the engine reads
    the files straight from the checkout (hasty_equity.cpp), no Macondo build
    involved."""
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


def fetch_kill_test_deps():
    """Everything a kill-test worker needs beyond the bundle itself."""
    fetch_lexicon(DEFAULT_LEXICON)
    fetch_macondo_strategy(DEFAULT_LEXICON)
