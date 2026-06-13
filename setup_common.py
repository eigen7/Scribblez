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

from subtrees.devenv_utils import DevenvConfig

REPO_ROOT = Path(__file__).resolve().parent

# Bumped manually whenever the Dockerfile changes in a way that requires users
# to rebuild. Checked at run_docker.py launch time against the running image's
# `version` label.
MINIMUM_REQUIRED_IMAGE_VERSION = "0.3.0"

# Ports forwarded host -> container by run_docker.py.
REQUIRED_PORTS = [
    5173,  # Vite dev server (browser UI)
    8080,  # engine WebSocket (default human_web_agent --port)
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


def make_config() -> DevenvConfig:
    """Build the Scribblez DevenvConfig consumed by every host-side script."""
    return DevenvConfig(
        name="scribblez",
        repo_root=REPO_ROOT,
        required_ports=REQUIRED_PORTS,
        min_image_version=MINIMUM_REQUIRED_IMAGE_VERSION,
    )
