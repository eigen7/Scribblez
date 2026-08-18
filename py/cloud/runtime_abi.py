"""The runtime ABI a compiled bundle needs from the machine that runs it.

Bundles carry code and binaries; the worker image carries the libraries they
link against. Those libraries come from the dev image -- copied out of it
(libstdc++, the NVIDIA runtime) or apt-installed to match it -- so the two
images are a matched pair, and rebuilding one without the other produces
binaries no worker can load:

    OSError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.35'
    not found (required by .../libscribblez_ffi.so)

which is what a gcc upgrade in the dev image did in August 2026. The pair is
kept in step by build_docker_image.py, which builds both; this module is the
check for when something bypasses that. The worker image's versions are
recorded at push time (build_and_push_worker_image.py) into the shared mount,
where the dev container can compare them against its own before deploying a
bundle built there.

Versions are read as the file behind each soname symlink ("libstdc++.so.6" ->
"libstdc++.so.6.0.35"), which is available on both sides without a compiler,
a package manager, or docker.
"""

import json
from pathlib import Path

# Where a tracked library may live. The CUDA runtime sits under the toolkit in
# the dev image and beside the rest in the worker image (which copies it
# there), so both are searched and the first hit wins.
LIB_DIRS = (Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/local/cuda/lib64"))

# The libraries whose version has to agree. libstdc++/libgcc_s move with the
# compiler and are backward compatible, so the worker's may not be older than
# the dev image's; the NVIDIA runtime is copied verbatim and version-locked to
# the TensorRT the engine was built against, so it must match exactly.
AT_LEAST = ("libstdc++.so.6", "libgcc_s.so.1")
EXACTLY = ("libnvinfer.so.10", "libcudart.so.12")

# Where the push records what it built, under the shared mount root.
RECORD_REL = "cloud/worker_image.json"


def _version(soname: str, lib_dirs) -> str:
    """The versioned file behind `soname`, the soname itself when it is not a
    symlink (libgcc_s ships as a real file on some images), or "" when this
    filesystem does not have it at all."""
    for lib_dir in lib_dirs:
        path = lib_dir / soname
        if path.exists():
            return path.resolve().name
    return ""


def local_versions(lib_dirs=LIB_DIRS) -> dict[str, str]:
    """What this filesystem provides, for every tracked library."""
    return {name: _version(name, lib_dirs) for name in AT_LEAST + EXACTLY}


def probe_command() -> list[str]:
    """A shell command printing what an image provides, in parse_versions'
    format. Lets the host ask the worker image directly -- it holds no repo to
    import this module from."""
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
    """The tracked libraries on which `worker` cannot run code built against
    `dev`: an older backward-compatible one, or a mismatched locked one. An
    empty result means a bundle from `dev` will load."""
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


def write_record(mount_root: Path, image: str, versions: dict[str, str]):
    """Record what the worker image now published provides."""
    path = record_path(mount_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"image": image, "versions": versions}, indent=2) + "\n")


def read_record(mount_root: Path) -> dict | None:
    """The last push's record, or None if no push has written one yet (in
    which case nothing can be said about the image and nothing is claimed)."""
    try:
        return json.loads(record_path(mount_root).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
