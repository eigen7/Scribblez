"""Shared helpers for setup_wizard.py / build_docker_image.py / run_docker.py.

Lives at the repo root (not under python/) so the host-side scripts can import
it without depending on PYTHONPATH or any in-container Python paths.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


# ---- Constants ----------------------------------------------------------

LOCAL_DOCKER_IMAGE = "scribblez"

# Name used by run_docker.py for the launched container; also the key under
# which we register a VS Code "Attach to Running Container" config.
DEFAULT_INSTANCE_NAME = "scribblez_instance"

# Bumped manually whenever the Dockerfile changes in a way that requires users
# to rebuild. Checked at run_docker.py launch time against the running image's
# `version` label.
MINIMUM_REQUIRED_IMAGE_VERSION = "0.1.0"

# Ports forwarded host -> container by run_docker.py.
REQUIRED_PORTS = [
    5173,  # Vite dev server (browser UI)
    8080,  # engine WebSocket (default human_web_agent --port)
]

# Paths inside the Docker container.
CONTAINER_REPO_PATH = "/workspace/repo"
CONTAINER_MOUNT_PATH = "/workspace/mount"

DIR = Path(__file__).resolve().parent
ENV_JSON_PATH = DIR / ".env.json"

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


# ---- .env.json ----------------------------------------------------------

def get_env_json() -> dict:
    if ENV_JSON_PATH.exists():
        with ENV_JSON_PATH.open() as f:
            return json.load(f)
    return {}


def update_env_json(mappings: dict) -> None:
    env = get_env_json()
    env.update(mappings)
    with ENV_JSON_PATH.open("w") as f:
        json.dump(env, f, indent=2)
        f.write("\n")


# ---- Misc helpers -------------------------------------------------------

Version = Tuple[int, ...]


def parse_version_str(version_str: str) -> Version:
    return tuple(int(x) for x in version_str.split("."))


def is_version_ok(version_str: str) -> bool:
    if not version_str:
        return False
    try:
        return parse_version_str(version_str) >= parse_version_str(
            MINIMUM_REQUIRED_IMAGE_VERSION
        )
    except ValueError:
        return False


def is_subpath(child: os.PathLike, parent: os.PathLike) -> bool:
    p = Path(child).resolve()
    d = Path(parent).resolve()
    return p.is_relative_to(d)


def in_docker_container() -> bool:
    # Set by the Dockerfile via `ENV DOCKER_IMAGE_VERSION=...`.
    return "DOCKER_IMAGE_VERSION" in os.environ


def get_image_label(image_name: str, label_key: str):
    """Return the value of a single LABEL on a local Docker image, or None."""
    import subprocess
    try:
        result = subprocess.check_output(
            [
                "docker", "inspect",
                f"--format={{{{index .Config.Labels \"{label_key}\"}}}}",
                image_name,
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    return result or None


# ---- VS Code "Attach to Running Container" config -----------------------
#
# Background: when you use the Dev Containers extension's
# "Attach to Running Container" command, VS Code does NOT consult
# .devcontainer/devcontainer.json in the workspace. Instead it reads a
# per-container config from the user's globalStorage:
#
#   <vscode-user-data>/User/globalStorage/
#       ms-vscode-remote.remote-containers/nameConfigs/<container>.json
#
# Without that file, VS Code has no way to know which user to attach as, and
# defaults to whatever USER the image declares -- which for our image is root
# (devuser is created at container start by entrypoint.sh, not at image-build
# time, so we can't bake USER devuser into the Dockerfile).
#
# setup_wizard.py writes this file at setup time so that attach Just Works.

# Subpath under each VS Code flavor's user-data dir.
_NAMECONFIGS_SUBPATH = Path(
    "User", "globalStorage", "ms-vscode-remote.remote-containers", "nameConfigs"
)


def _vscode_user_data_roots() -> list[Path]:
    """Candidate user-data roots for installed VS Code flavors on this OS.

    We return one path per flavor (Code, Code - Insiders, VSCodium) regardless
    of whether it actually exists; the caller decides what to do with missing
    directories. Order: stable, insiders, vscodium.
    """
    flavors = ["Code", "Code - Insiders", "VSCodium"]
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return []
        base = Path(appdata)
    else:
        # Linux / *BSD: honor XDG_CONFIG_HOME, default to ~/.config.
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return [base / flavor for flavor in flavors]


def vscode_attach_config_paths(instance_name: str) -> list[Path]:
    """Paths to nameConfigs/<instance>.json for every VS Code flavor present.

    A flavor is considered "present" if its user-data root directory already
    exists on disk (we don't want to materialize a config dir for an editor the
    user doesn't use).
    """
    paths: list[Path] = []
    for root in _vscode_user_data_roots():
        if root.is_dir():
            paths.append(root / _NAMECONFIGS_SUBPATH / f"{instance_name}.json")
    return paths


def desired_vscode_attach_config(instance_name: str) -> dict:
    """Minimal attach config we want to ensure is present."""
    return {
        "containerName": instance_name,
        "remoteUser": "devuser",
        "workspaceFolder": CONTAINER_REPO_PATH,
    }


def write_vscode_attach_config(path: Path, instance_name: str) -> str:
    """Create or merge the attach config at `path`. Returns a status string.

    - If `path` does not exist: write the desired config.
    - If `path` exists: merge our keys in, preserving any extra keys (e.g.
      `extensions`, `settings`) the user has added by hand. We overwrite our
      three keys if they differ, since their whole purpose is to make attach
      work correctly.

    Returns one of: "created", "updated", "unchanged".
    """
    desired = desired_vscode_attach_config(instance_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, dict):
                raise ValueError(f"top-level JSON in {path} is not an object")
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Refusing to overwrite {path}: it exists but is not valid JSON "
                f"({e}). Fix or delete it and re-run."
            )
        merged = dict(existing)
        merged.update(desired)
        if merged == existing:
            return "unchanged"
        path.write_text(json.dumps(merged, indent=4) + "\n")
        return "updated"

    path.write_text(json.dumps(desired, indent=4) + "\n")
    return "created"
