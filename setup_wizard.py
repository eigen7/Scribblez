#!/usr/bin/env python3
"""Interactive first-time setup for Scribblez.

Run this *outside* the Docker container. It:
  1. Picks a persistent host directory ("mount dir") to be bind-mounted into
     the container at /workspace/mount. Build artifacts that need to outlive a
     single container, plus large data files (Macondo, lexica), live there.
  2. Verifies you can run `docker` without sudo.
  3. Downloads .kwg lexicon files from the public Woogles/liwords repo into
     <mount>/lexica/, and symlinks them into Macondo's own data dir so the
     macondo subprocess can find them too. The KWG files are not redistributed
     by Scribblez; we just automate the same fetch-from-upstream the user
     would do by hand.
  4. Writes a per-container VS Code config so that "Dev Containers: Attach
     to Running Container" connects as devuser instead of root. (VS Code's
     attach flow ignores .devcontainer/devcontainer.json -- it reads a
     separate file under the user-data dir.)
  5. Optionally builds the Docker image (delegates to build_docker_image.py).

The Macondo checkout and binary are managed by build.py, not this wizard.

Re-run the wizard any time you want to install additional lexica, refresh the
VS Code attach config, or rebuild the image.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setup_common import (
    CONTAINER_MOUNT_PATH,
    DEFAULT_INSTANCE_NAME,
    DEFAULT_LEXICA,
    LIWORDS_KWG_URL_TEMPLATE,
    get_env_json,
    in_docker_container,
    is_subpath,
    update_env_json,
    vscode_attach_config_paths,
    write_vscode_attach_config,
)


REPO_ROOT = Path(__file__).resolve().parent


# ---- Pretty printing ----------------------------------------------------

def print_red(text: str) -> None:
    print(f"\033[31m{text}\033[0m")


def print_green(text: str) -> None:
    print(f"\033[32m{text}\033[0m")


class SetupException(Exception):
    """Caught at top level; only the first arg is printed (no traceback)."""


# ---- Step 1: mount dir --------------------------------------------------

def setup_mount_dir() -> Path:
    print("Scribblez needs a persistent directory on the host that gets")
    print(f"bind-mounted into the Docker container at {CONTAINER_MOUNT_PATH}.")
    print("It holds the Macondo checkout/binary, lexicon files, and any future")
    print("self-play data. It MUST live outside this repo.")
    print()

    env = get_env_json()
    default = env.get("MOUNT_DIR", os.path.expanduser("~/scribblez-mount"))

    while True:
        ans = input(f"Mount directory [{default}]: ").strip() or default
        target = os.path.abspath(os.path.expanduser(ans))
        if is_subpath(target, REPO_ROOT):
            print_red(f"Mount dir cannot live inside the repo ({REPO_ROOT}).")
            continue
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as e:
            print_red(f"Could not create {target}: {e}")
            continue
        break

    update_env_json({"MOUNT_DIR": target})
    print_green(f"Mount dir: {target}")
    return Path(target)


# ---- Step 2: docker permissions -----------------------------------------

def check_docker_permissions() -> None:
    print("Checking that you can run `docker` without sudo...")
    result = subprocess.run(
        ["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode == 0:
        print_green("Docker is usable without sudo.")
        return

    if "permission denied" in result.stderr.lower():
        print_red("You can't run docker without sudo. Add yourself to the docker group:")
        print("    sudo usermod -aG docker $USER")
        print("Then log out and back in.")
    else:
        print_red("`docker ps` failed:")
        print(result.stderr)
    raise SetupException()


# ---- helpers -----------------------------------------------------------

def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ---- Step 3: lexica -----------------------------------------------------

def installed_lexica(mount: Path) -> list[str]:
    lex_dir = mount / "lexica"
    if not lex_dir.is_dir():
        return []
    return sorted(p.stem for p in lex_dir.glob("*.kwg"))


def _download(url: str, dest: Path) -> bool:
    """Download `url` to `dest` atomically. Prefer curl, fall back to wget."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if _have("curl"):
        cmd = ["curl", "-fL", "--retry", "3", "-o", str(tmp), url]
    elif _have("wget"):
        cmd = ["wget", "-q", "-O", str(tmp), url]
    else:
        print_red("Need `curl` or `wget` to download lexica.")
        return False
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.rename(dest)
    return True


def setup_lexica(mount: Path) -> None:
    """Interactive: list installed lexica, prompt for additional ones to fetch.

    Also keeps Macondo's expected data/lexica/gaddag/ dir in sync via symlinks
    (Macondo looks up its kwg by lexicon name from there).
    """
    lex_dir = mount / "lexica"
    macondo_lex_dir = mount / "macondo" / "data" / "lexica" / "gaddag"
    lex_dir.mkdir(parents=True, exist_ok=True)
    macondo_lex_dir.mkdir(parents=True, exist_ok=True)

    have = installed_lexica(mount)
    if have:
        print(f"Already installed lexica ({len(have)}): {', '.join(have)}")
    else:
        print("No lexica installed yet.")

    proposed = [name for name in DEFAULT_LEXICA if name not in have]
    if proposed:
        default_csv = ",".join(proposed)
    else:
        default_csv = ""

    prompt = (
        "Lexica to install (comma-separated names; blank to skip)"
        f"{f' [{default_csv}]' if default_csv else ''}: "
    )
    ans = input(prompt).strip()
    if not ans:
        ans = default_csv
    if not ans:
        print("Skipping lexicon install.")
        return

    requested = [s.strip() for s in ans.split(",") if s.strip()]
    failed = []
    for name in requested:
        if name in installed_lexica(mount):
            print(f"  {name}: already installed; skipping.")
        else:
            url = LIWORDS_KWG_URL_TEMPLATE.format(name=name)
            print(f"Fetching {name} ...")
            if not _download(url, lex_dir / f"{name}.kwg"):
                print_red(f"  Failed to download {name}.")
                failed.append(name)
                continue

        # Keep macondo's data/lexica/gaddag/<name>.kwg in sync with our copy.
        link = macondo_lex_dir / f"{name}.kwg"
        target = lex_dir / f"{name}.kwg"
        if link.is_symlink() or link.exists():
            link.unlink()
        # Use a relative symlink so the link stays valid even if the user
        # eventually moves the mount dir.
        link.symlink_to(os.path.relpath(target, link.parent))

    print()
    have = installed_lexica(mount)
    print_green(f"Installed lexica ({len(have)}): {', '.join(have) or '(none)'}")
    if failed:
        print_red(f"Failed: {', '.join(failed)}")


# ---- Step 4: VS Code "Attach to Running Container" config ---------------

def _yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default_yes
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def setup_vscode_attach_config() -> None:
    """Write a per-container config so VS Code's Dev Containers extension
    attaches as `devuser` (not root) when the user runs
    "Dev Containers: Attach to Running Container".
    """
    paths = vscode_attach_config_paths(DEFAULT_INSTANCE_NAME)
    if not paths:
        print("No VS Code user-data directory found for any flavor")
        print("(Code / Code - Insiders / VSCodium). Skipping VS Code attach")
        print("config. If you install VS Code later, re-run this wizard, or")
        print("manually use 'Dev Containers: Open Named Container Configuration")
        print("File' and set \"remoteUser\": \"devuser\".")
        return

    print("VS Code's 'Attach to Running Container' command does NOT read")
    print("this repo's .devcontainer/devcontainer.json. It reads a per-container")
    print("config under the VS Code user-data dir. Without it, vscode-server")
    print("runs as root inside the container.")
    print()
    print("Detected user-data dir(s); proposing to write/merge:")
    for p in paths:
        print(f"  {p}")
    print()
    if not _yes_no("Configure VS Code attach to run as devuser?"):
        print("Skipping VS Code attach config.")
        return

    for p in paths:
        try:
            status = write_vscode_attach_config(p, DEFAULT_INSTANCE_NAME)
        except RuntimeError as e:
            print_red(str(e))
            continue
        if status == "created":
            print_green(f"Wrote {p}")
        elif status == "updated":
            print_green(f"Updated {p} (merged in remoteUser/workspaceFolder/containerName)")
        else:
            print(f"{p} already up to date.")


# ---- Step 5: build the Docker image -------------------------------------

def build_docker_image() -> None:
    if not _yes_no("Build the Scribblez Docker image now (./build_docker_image.py)?"):
        print("Skipping image build. Run ./build_docker_image.py before ./run_docker.py.")
        return
    # Import lazily so a broken build_docker_image.py doesn't prevent the
    # earlier wizard steps from running.
    import build_docker_image as bdi
    from setup_common import LOCAL_DOCKER_IMAGE
    rc = bdi.docker_build(LOCAL_DOCKER_IMAGE)
    if rc != 0:
        raise SetupException("Docker image build failed.")


# ---- Driver -------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser.parse_args()


def main() -> None:
    assert not in_docker_container(), \
        "setup_wizard.py is intended to be run on the host, not inside the container."
    get_args()  # for --help

    print("*" * 78)
    print("Scribblez setup wizard")
    print("*" * 78)
    os.chdir(REPO_ROOT)

    try:
        mount = setup_mount_dir()
        print("*" * 78)
        check_docker_permissions()
        print("*" * 78)
        setup_lexica(mount)
        print("*" * 78)
        setup_vscode_attach_config()
        print("*" * 78)
        build_docker_image()
        print("*" * 78)
        print_green("Setup complete.")
        print("Next: ./run_docker.py")
    except KeyboardInterrupt:
        print()
        print("Setup wizard interrupted. Re-run when ready.")
        sys.exit(1)
    except SetupException as e:
        for arg in e.args:
            print("*" * 78)
            print(arg)
        sys.exit(1)


if __name__ == "__main__":
    main()
