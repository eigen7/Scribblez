"""Project-specific devenv configuration and data-fetch constants for Scribblez.

The generic host-side machinery (Docker build/run, .env.json, VS Code attach
config, NVIDIA validation, the setup-wizard scaffold) lives in the vendored
`subtrees/devenv_utils` copy. The static DevenvConfig fields live as data in
the repo-root `devenv.toml`; this module loads them and adds the
Scribblez-specific lexica/Macondo constants used by the custom wizard steps.

It lives at the repo root (not under py/) so the host-side scripts can
import it without depending on PYTHONPATH or any in-container Python paths.
"""

from pathlib import Path

from subtrees.devenv_utils import (
    DevenvConfig,
    DevTool,
    load_config,
)
from subtrees.devenv_utils import (
    check_setup_version as _check_setup_version,
)

REPO_ROOT = Path(__file__).resolve().parent

# Lexica we know how to fetch. The KWG files are not in this repo; they are
# downloaded at setup time from the public liwords repo into the user's mount
# directory. We never redistribute them ourselves. The URL is pinned to the
# last liwords commit that has them: liwords deleted wasm/2024/ on 2026-11-19
# (3f606be4b7, "Remove the 2024 analyzer files"), and a moving branch would
# break again or change the files under us.
LIWORDS_KWG_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/woogles-io/liwords/9315e18a78d3854debed8b90d1c3618abd91ee08/"
    "liwords-ui/public/wasm/2024/{name}.kwg"
)

# Default set proposed during the "install lexica" wizard step. NWL23 is the
# Scribblez default; the others are common alternatives.
DEFAULT_LEXICA = ["NWL23", "NWL20", "CSW24", "NSWL23"]

MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"


def check_setup_version():
    _check_setup_version(make_config())


def dev_tool() -> DevTool:
    """Return the project's dev-workflow helper (clang-format)."""
    return DevTool(make_config())


def make_config() -> DevenvConfig:
    """The Scribblez DevenvConfig consumed by every host-side script, loaded
    from the repo-root devenv.toml."""
    return load_config(REPO_ROOT)
