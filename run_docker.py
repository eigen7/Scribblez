#!/usr/bin/env python3
"""Launch (or attach to) the Scribblez Docker container.

Mounts:
  <repo>       -> /workspace/repo   (your live source tree; build artifacts in
                                     build/ persist on the host)
  <mount-dir>  -> /workspace/mount  (Macondo checkout, lexica, future data)

Forwards the ports the engine's web UI uses (Vite + WebSocket).

Drops you into a bash shell inside the container as `devuser`, whose UID/GID
match your host user so that anything written into the bind-mounts is owned
by you on the host.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from setup_common import (
    DEFAULT_INSTANCE_NAME,
    LOCAL_DOCKER_IMAGE,
    MINIMUM_REQUIRED_IMAGE_VERSION,
    REQUIRED_PORTS,
    get_env_json,
    get_image_label,
    is_subpath,
    is_version_ok,
)


REPO_ROOT = Path(__file__).resolve().parent


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--docker-image",
                        help=f"image to run (default: {LOCAL_DOCKER_IMAGE})")
    parser.add_argument("-i", "--instance-name", default=DEFAULT_INSTANCE_NAME,
                        help="container name (default: %(default)s)")
    parser.add_argument("-s", "--skip-image-version-check", action="store_true",
                        help="skip the image-version label check")
    return parser.parse_args()


def is_container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def check_image_version(image: str) -> bool:
    version = get_image_label(image, "version")
    if is_version_ok(version):
        return True
    if version is None:
        print("Image is missing a `version` label; it may be out of date.")
    else:
        print(f"Image version {version} < required {MINIMUM_REQUIRED_IMAGE_VERSION}.")
    print("Run ./build_docker_image.py to rebuild, "
          "or pass --skip-image-version-check.")
    return False


def exec_into_running(name: str) -> None:
    cmd = ["docker", "exec", "-it", name, "gosu", "devuser", "bash"]
    print(f"$ {' '.join(shlex.quote(a) for a in cmd)}")
    subprocess.run(cmd, check=False)


def run_container(args: argparse.Namespace) -> None:
    env = get_env_json()
    image = args.docker_image or env.get("DOCKER_IMAGE") or LOCAL_DOCKER_IMAGE
    mount_dir = env.get("MOUNT_DIR")
    if not mount_dir:
        print("Error: MOUNT_DIR is not set. Run ./setup_wizard.py first.")
        sys.exit(1)
    if not os.path.isdir(mount_dir):
        print(f"Error: mount dir {mount_dir} does not exist. Re-run ./setup_wizard.py.")
        sys.exit(1)
    assert not is_subpath(mount_dir, REPO_ROOT), \
        f"Mount dir {mount_dir} must not live inside repo {REPO_ROOT}"

    if not args.skip_image_version_check and not check_image_version(image):
        sys.exit(1)

    uid = subprocess.check_output(["id", "-u"], text=True).strip()
    gid = subprocess.check_output(["id", "-g"], text=True).strip()

    cmd = [
        "docker", "run", "--rm", "-it",
        "--name", args.instance_name,
        "-e", f"HOST_UID={uid}",
        "-e", f"HOST_GID={gid}",
        "-e", "USERNAME=devuser",
        "-v", f"{REPO_ROOT}:/workspace/repo",
        "-v", f"{mount_dir}:/workspace/mount",
    ]
    for port in REQUIRED_PORTS:
        cmd += ["-p", f"{port}:{port}"]
    cmd += [image, "bash"]

    print(f"$ {' '.join(shlex.quote(a) for a in cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    args = get_args()
    if is_container_running(args.instance_name):
        print(f"Container {args.instance_name} already running; exec'ing in.")
        exec_into_running(args.instance_name)
    else:
        run_container(args)


if __name__ == "__main__":
    main()
