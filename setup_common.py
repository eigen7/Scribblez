"""Project-specific devenv configuration and data-fetch constants for Scribblez.

The generic host-side machinery (Docker build/run, .env.json, VS Code attach
config, NVIDIA validation, the setup-wizard scaffold) lives in the
`subtrees/devenv_utils` git submodule. This module supplies the Scribblez-specific
pieces: a `DevenvConfig` factory and the lexica/Macondo constants used by the
custom wizard steps.

It lives at the repo root (not under py/) so the host-side scripts can
import it without depending on PYTHONPATH or any in-container Python paths.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# subtrees/devenv_utils is a git submodule, so a clone made without
# --recurse-submodules leaves its directory empty. Every host-side entry point
# imports this module before anything else, so populating the submodule here --
# before the imports below -- makes a plain `git clone` just work.
if not (REPO_ROOT / "subtrees" / "devenv_utils" / "__init__.py").exists():
    subprocess.run(["git", "submodule", "update", "--init"], cwd=REPO_ROOT, check=True)

from subtrees.devenv_utils import (  # noqa: E402
    DevenvConfig,
    DevTool,
)
from subtrees.devenv_utils import (  # noqa: E402
    check_setup_version as _check_setup_version,
)

# Bump SETUP_VERSION manually to force all users to rerun the setup wizard.
#
# Increasing the major version (the first number) causes the setup wizard to
# rm -rf the target/ directory - use this to invalidate existing builds.
SETUP_VERSION = "2.10.0"

# Bumped manually whenever the Dockerfile changes in a way that requires users
# to rebuild. Checked at run_docker.py launch time against the running image's
# `version` label.
MINIMUM_REQUIRED_IMAGE_VERSION = "3.7.0"

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

# Gitea (local git forge for worktree-branch PR review; see
# py/tools/gitea_serve.py) is published outside REQUIRED_PORTS so it binds
# 127.0.0.1 on the host instead of 0.0.0.0: its nginx front-end signs every
# request in as the admin user, so the port must not be reachable from the
# LAN. Note the fixed host port assumes a single container instance.
GITEA_PORT_ARGS = ["-p", "127.0.0.1:3000:3000"]

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


def check_setup_version():
    _check_setup_version(make_config())


def dev_tool() -> DevTool:
    """Return the project's dev-workflow helper (clang-format)."""
    return DevTool(make_config())


def make_config() -> DevenvConfig:
    """Build the Scribblez DevenvConfig consumed by every host-side script."""
    return DevenvConfig(
        name="scribblez",
        repo_root=REPO_ROOT,
        docker_context=REPO_ROOT / "docker-setup" / "local",
        required_ports=REQUIRED_PORTS,
        min_image_version=MINIMUM_REQUIRED_IMAGE_VERSION,
        setup_version=SETUP_VERSION,
        extra_docker_args=GITEA_PORT_ARGS,
    )
