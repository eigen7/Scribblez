"""Which worker image a role runs on, and a check that the image can load
bundles built in this dev container.

Bundles carry code and binaries; the worker images carry the shared libraries
those binaries link against. The libraries come from the dev image (copied out
of it, or apt-installed to match), so the dev image and the worker images are
a matched set. Upgrading the dev image's compiler without rebuilding the
worker images (build_and_push_worker_image.py, run by hand) produces bundles
no worker can load:

    OSError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.35'
    not found (required by .../libscribblez_ffi.so)

To catch this before deploying, the image push records each image's library
versions under the shared mount, and the dashboard compares them against the
dev container's own (stale_libraries) before it deploys a bundle.

There is one image per *runtime* a role declares (RoleSpec.runtime):

  engine  the engine binaries and the ctypes FFI: numpy, the C++ and NVIDIA
          runtime libraries, TensorRT's builder. Generators and match eval.
  torch   the engine runtime plus PyTorch and the training stack, for train
          roles. It is a later stage of the same Dockerfile, so it has every
          library the engine image has, at the same versions.

A library's version is the name of the file its soname symlink resolves to
("libstdc++.so.6" -> "libstdc++.so.6.0.35"). That can be read on either side
without a compiler, a package manager or docker.
"""

import json
from pathlib import Path

RUNTIME_ENGINE = "engine"
RUNTIME_TORCH = "torch"
RUNTIMES = (RUNTIME_ENGINE, RUNTIME_TORCH)

# The docker-setup/worker/Dockerfile stage that builds each runtime's image,
# and the tag suffix that distinguishes the torch image from the engine one
# (see RegistryConfig.image_for in cloud/credentials.py).
DOCKER_TARGET = {RUNTIME_ENGINE: "worker", RUNTIME_TORCH: "worker-torch"}
TORCH_TAG_SUFFIX = "-torch"

# Where a tracked library may live. The dev image keeps the CUDA runtime under
# the toolkit; the worker image copies it beside the other libraries. The
# first hit wins.
LIB_DIRS = (Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/local/cuda/lib64"))

# The libraries whose versions must agree. libstdc++ and libgcc_s are
# backward compatible, so the worker's need only be at least as new as the dev
# image's. The NVIDIA libraries are locked to the TensorRT the engine was built
# against, so they must match exactly.
AT_LEAST = ("libstdc++.so.6", "libgcc_s.so.1")
EXACTLY = ("libnvinfer.so.10", "libcudart.so.12")

# The push's record, relative to the shared mount root. Format:
# {"images": {runtime: {"image": name, "versions": {soname: file}}}}.
RECORD_REL = "cloud/worker_image.json"


def _version(soname: str, lib_dirs) -> str:
    """The versioned file behind `soname`; the soname itself when it is a
    real file rather than a symlink (libgcc_s, on some images); "" when
    absent."""
    for lib_dir in lib_dirs:
        path = lib_dir / soname
        if path.exists():
            return path.resolve().name
    return ""


def local_versions(lib_dirs=LIB_DIRS) -> dict[str, str]:
    """The tracked libraries' versions on this filesystem."""
    return {name: _version(name, lib_dirs) for name in AT_LEAST + EXACTLY}


def probe_command() -> list[str]:
    """A shell equivalent of local_versions, printing parse_versions' format.
    The image push runs it inside a worker image, which has no repo to import
    this module from."""
    dirs = " ".join(str(d) for d in LIB_DIRS)
    names = " ".join(AT_LEAST + EXACTLY)
    return [
        "sh",
        "-c",
        f"for n in {names}; do for d in {dirs}; do "
        '[ -e "$d/$n" ] && printf \'%s %s\\n\' "$n" "$(basename "$(readlink -f "$d/$n")")" '
        "&& break; done; done",
    ]


def parse_versions(text: str) -> dict[str, str]:
    """The versions in probe_command's output."""
    pairs = (line.split() for line in text.splitlines() if line.strip())
    return {name: version for name, version in pairs}


def _ordering(version: str) -> list[int]:
    return [int(part) for part in version.split(".") if part.isdigit()]


def stale_libraries(worker: dict, dev: dict) -> list[str]:
    """The libraries that stop `worker` from loading a bundle built against
    `dev`: a backward-compatible one that is older, or a locked one that
    differs. Empty means the bundle will load."""
    stale = []
    for name in AT_LEAST:
        if _ordering(worker.get(name, "")) < _ordering(dev.get(name, "")):
            stale.append(name)
    for name in EXACTLY:
        if dev.get(name) and worker.get(name, "") != dev[name]:
            stale.append(name)
    return stale


def record_path(mount_root: Path) -> Path:
    return Path(mount_root) / RECORD_REL


def write_record(mount_root: Path, runtime: str, image: str, versions: dict[str, str]):
    """Record the versions in the just-pushed `runtime` image, keeping the
    other runtimes' entries."""
    assert runtime in RUNTIMES, runtime
    path = record_path(mount_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (read_records(mount_root) or {}) | {runtime: {"image": image, "versions": versions}}
    path.write_text(json.dumps({"images": images}, indent=2) + "\n")


def read_records(mount_root: Path) -> dict[str, dict] | None:
    """Each runtime's last push, runtime -> {image, versions}, or None when
    no push has been recorded (the check then has nothing to compare)."""
    try:
        return json.loads(record_path(mount_root).read_text())["images"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None
