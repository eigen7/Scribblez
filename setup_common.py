"""Project-specific devenv configuration and data-fetch constants for Scribblez.

The generic host-side machinery (Docker build/run, .env.json, VS Code attach
config, NVIDIA validation, the setup-wizard scaffold) lives in the
`subtrees/devenv_utils` git subtree. This module supplies the Scribblez-specific
pieces: a `DevenvConfig` factory and the lexica/Macondo constants used by the
custom wizard steps.

It lives at the repo root (not under py/) so the host-side scripts can
import it without depending on PYTHONPATH or any in-container Python paths.
"""

from pathlib import Path

from subtrees.devenv_utils import (
    DevenvConfig,
    DevTool,
    SubtreeSpec,
)
from subtrees.devenv_utils import (
    check_setup_version as _check_setup_version,
)

REPO_ROOT = Path(__file__).resolve().parent

# Bump SETUP_VERSION manually to force all users to rerun the setup wizard.
#
# Increasing the major version (the first number) causes the setup wizard to
# rm -rf the target/ directory - use this to invalidate existing builds.
SETUP_VERSION = "2.6.1"

# Bumped manually whenever the Dockerfile changes in a way that requires users
# to rebuild. Checked at run_docker.py launch time against the running image's
# `version` label.
MINIMUM_REQUIRED_IMAGE_VERSION = "3.4.0"

# Ports forwarded host -> container by run_docker.py.
REQUIRED_PORTS = [
    5173,  # Vite dev server (browser UI)
    5174,  # Vite dev server (browser UI)
    5175,  # Vite dev server (browser UI)
    8080,  # engine WebSocket (default human_web_agent --port)
    5006,  # Bokeh training-metrics dashboard
    5180,  # React dashboard: Vite dev server (the page the browser opens)
    8090,  # React dashboard: Tornado data API (proxied by Vite; handy for direct access)
]

# Lexica we know how to fetch. The KWG files are not in this repo; they are
# downloaded at setup time from the public liwords URL into the user's mount
# directory. We never redistribute them ourselves.
LIWORDS_KWG_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/woogles-io/liwords/master/"
    "liwords-ui/public/wasm/2024/{name}.kwg"
)

# Default set proposed during the "install lexica" wizard step. NWL23 is the
# Scribblez default; the others are common alternatives.
DEFAULT_LEXICA = ["NWL23", "NWL20", "CSW24", "NSWL23"]

MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"

# Git subtrees vendored under subtrees/. git records neither each subtree's
# remote URL nor its tracked branch, so they are declared here and consumed by
# the pull/push tools in py/tools/ via DevTool.
SUBTREES = [
    SubtreeSpec(name="devenv_utils", url="https://github.com/eigen7/devenv_utils.git"),
]


def check_setup_version():
    _check_setup_version(make_config())


def dev_tool() -> DevTool:
    """Return the project's dev-workflow helper (clang-format, git subtrees)."""
    return DevTool(make_config())


def make_config() -> DevenvConfig:
    """Build the Scribblez DevenvConfig consumed by every host-side script."""
    return DevenvConfig(
        name="scribblez",
        repo_root=REPO_ROOT,
        required_ports=REQUIRED_PORTS,
        min_image_version=MINIMUM_REQUIRED_IMAGE_VERSION,
        setup_version=SETUP_VERSION,
        subtrees=SUBTREES,
    )
