#!/usr/bin/env python3
"""Build Scribblez: compile the C++ engine and install the web UI's npm deps.

This is the one command you run before playing. The build step installs the
front-end's npm packages, and the engine (play_game) launches the Vite dev
server itself at play time -- so you never invoke npm by hand.

Usage:
    py/build.py [--debug] [--clean] [-j N]

Then play a human-vs-AI game with:
    ./target/engine/play_game --player "--type=human" --player "--type=greedy"
"""

import argparse
import os
import shutil
import subprocess
import sys

from scribblez.hardware import default_thread_count
from setup_check import import_setup_common
from util.argparse_ext import ArgumentDefaultsHelpFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(ROOT, "target")
ARCHS_DIR = os.path.join(TARGET_DIR, "archs")

# CPU microarchitectures (GCC/Clang -march values) this project builds
# bundles for, e.g. for cloud tooling to upload one binary bundle per arch and
# have each worker fetch the one matching its CPU. Grows by hand: when
# building on a host whose arch isn't listed here, build_engine() below warns
# so the operator can add it and commit.
SUPPORTED_ARCHS = ["alderlake", "x86-64", "znver2", "znver4"]

# Pinned Macondo release. build.py will clone this tag if the repo is absent,
# and will error if the existing checkout is at a different tag (unless
# --skip-macondo-tag-check is passed).
MACONDO_TAG = "v0.13.1"
MOUNT_DIR = "/workspace/mount"
MACONDO_DIR = os.path.join(MOUNT_DIR, "macondo")
MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"

# Lexicon .kwg files are downloaded into <mount>/lexica/ by setup_wizard.py.
# Macondo expects to find them under <macondo>/data/lexica/gaddag/, keyed by
# lexicon name; we point that directory at <mount>/lexica so the macondo
# subprocess can resolve them.
LEXICA_DIR = os.path.join(MOUNT_DIR, "lexica")
MACONDO_GADDAG_DIR = os.path.join(MACONDO_DIR, "data", "lexica", "gaddag")


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def detect_host_arch() -> str:
    """Return this host's CPU microarchitecture as a GCC/Clang -march value.

    Asks the compiler what "-march=native" resolves to (e.g. "alderlake",
    "znver3") instead of inspecting /proc/cpuinfo directly, so the returned
    string is always one the local compiler accepts as an -march value.
    """
    result = subprocess.run(
        ["g++", "-march=native", "-Q", "--help=target"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("-march="):
            arch = line.split("=", 1)[1].strip()
            if arch and arch != "native":
                return arch
    sys.exit("Could not determine host CPU arch via `g++ -march=native -Q --help=target`.")


def arch_build_dir(arch: str) -> str:
    return os.path.join(ARCHS_DIR, arch)


def build_engine(arch: str, build_type: str, jobs: int):
    """Configure + compile the C++ engine for one CPU microarchitecture."""
    build_dir = arch_build_dir(arch)
    print(f"\nBuilding engine for arch '{arch}' ({build_type}) in {build_dir} ...")
    run(f"cmake -S . -B {build_dir} -DCMAKE_BUILD_TYPE={build_type} -DSCRIBBLEZ_MARCH={arch}")
    run(f"cmake --build {build_dir} -j{jobs}")


def _replace_with_symlink(link_path: str, real_path: str):
    """Point `link_path` at `real_path` (relative symlink), replacing whatever
    was already there -- a stale real file/dir left over from a single-arch
    build, or a symlink from a previously active arch."""
    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        shutil.rmtree(link_path)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(os.path.relpath(real_path, os.path.dirname(link_path)), link_path)


def link_host_arch_build(arch: str):
    """Point target/engine and target/compile_commands.json at this host
    arch's build under target/archs/, so tooling that hardcodes those two
    paths (tests, scripts, the FFI loader, clangd) keeps working regardless of
    which archs were actually built.
    """
    build_dir = arch_build_dir(arch)
    _replace_with_symlink(os.path.join(TARGET_DIR, "engine"), os.path.join(build_dir, "engine"))
    _replace_with_symlink(
        os.path.join(TARGET_DIR, "compile_commands.json"),
        os.path.join(build_dir, "compile_commands.json"),
    )
    rel_build_dir = os.path.relpath(build_dir, ROOT)
    print(f"\nLinked target/engine, target/compile_commands.json -> {rel_build_dir}")


def list_built_binaries(target_dir: str) -> list[str]:
    engine_dir = os.path.join(target_dir, "engine")
    binaries = []
    for name in os.listdir(engine_dir) if os.path.isdir(engine_dir) else []:
        path = os.path.join(engine_dir, name)
        if os.path.splitext(name)[1] == "" and os.path.isfile(path) and os.access(path, os.X_OK):
            binaries.append(path)
    return sorted(binaries)


def print_built_binaries(target_dir: str):
    binaries = list_built_binaries(target_dir)
    if not binaries:
        print("\nNo runnable binaries found under target/.")
        return
    print(f"\nBuilt {len(binaries)} binaries under target/:")
    for path in binaries:
        rel = os.path.relpath(path, ROOT)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    {rel}  ({size_mb:.1f} MB)")


def link_lexica_into_macondo():
    """Point Macondo's data/lexica/gaddag dir at <mount>/lexica.

    Macondo looks up its kwg by lexicon name from there. Symlinking the whole
    directory (rather than each .kwg) makes every installed lexicon resolvable
    and picks up lexica added later without re-linking. Uses a relative symlink
    so it survives the mount dir being moved. Safe to run repeatedly.
    """
    if not os.path.isdir(LEXICA_DIR):
        print(
            f"\nNo lexica dir at {LEXICA_DIR}; skipping macondo lexica link.\n"
            "Run ./setup_wizard.py to install lexica."
        )
        return
    parent = os.path.dirname(MACONDO_GADDAG_DIR)
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(MACONDO_GADDAG_DIR):
        os.unlink(MACONDO_GADDAG_DIR)
    elif os.path.isdir(MACONDO_GADDAG_DIR):
        shutil.rmtree(MACONDO_GADDAG_DIR)
    os.symlink(os.path.relpath(LEXICA_DIR, parent), MACONDO_GADDAG_DIR)
    print(f"\nLinked {LEXICA_DIR} -> {MACONDO_GADDAG_DIR}")


def clone_and_build_macondo():
    """Clone the pinned Macondo tag and build its shell binary, atomically.

    The clone and `go build` are coupled so the build runs only once, when the
    checkout is first created -- not on every build.py run. If either step
    fails, the partial checkout is removed so the next run starts clean.
    """
    print(f"\nCloning Macondo {MACONDO_TAG} into {MACONDO_DIR} ...")
    try:
        run(
            f"git -c advice.detachedHead=false clone --branch {MACONDO_TAG} --depth 1 --quiet "
            f"{MACONDO_REPO_URL} {MACONDO_DIR}"
        )
        os.makedirs(os.path.join(MACONDO_DIR, "bin"), exist_ok=True)
        run("go build -o bin/shell ./cmd/shell", cwd=MACONDO_DIR)
    except BaseException:
        shutil.rmtree(MACONDO_DIR, ignore_errors=True)
        raise


def check_macondo_tag():
    """Error out if the existing Macondo checkout is not at the expected tag."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
        cwd=MACONDO_DIR,
    )
    current_tag = result.stdout.strip()
    if result.returncode != 0 or current_tag != MACONDO_TAG:
        display = current_tag or "(not on an exact tag)"
        print(
            f"\nError: macondo at {MACONDO_DIR} is at '{display}', "
            f"but this project expects '{MACONDO_TAG}'.\n"
            "Update MACONDO_TAG in py/build.py, re-clone the directory, "
            "or pass --skip-macondo-tag-check to build anyway."
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="debug build (default: Release)")
    parser.add_argument(
        "--clean", action="store_true", help="remove the target/ directory before configuring"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=0, help="parallel build jobs (default: all CPUs)"
    )
    parser.add_argument(
        "-b",
        "--build-for-all-archs",
        action="store_true",
        help="build once per entry in SUPPORTED_ARCHS (py/build.py) instead of "
        "just this host's arch, e.g. to produce cloud-distributable bundles",
    )
    parser.add_argument(
        "--skip-web", action="store_true", help="skip installing the web UI npm dependencies"
    )
    parser.add_argument(
        "--skip-macondo", action="store_true", help="skip rebuilding the Macondo shell binary"
    )
    parser.add_argument(
        "--skip-macondo-tag-check",
        action="store_true",
        help="skip verifying that the macondo checkout is at "
        f"the expected tag ({MACONDO_TAG}); useful when "
        "fiddling with the macondo source",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    import_setup_common().check_setup_version()

    target_dir = TARGET_DIR
    if args.clean and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    elif os.path.exists(os.path.join(target_dir, "CMakeCache.txt")) and not os.path.isdir(
        ARCHS_DIR
    ):
        # Pre-multi-arch layout: a single CMake build sat directly under
        # target/. Its cache paths don't match the new target/archs/<arch>
        # layout, so it can't be reused -- wipe it and reconfigure fresh.
        print(f"\nFound a pre-multi-arch build at {target_dir}; removing it to reconfigure.")
        shutil.rmtree(target_dir)

    # 1. Configure + compile the C++ engine, once per requested arch.
    build_type = "Debug" if args.debug else "Release"
    jobs = args.jobs or default_thread_count()
    host_arch = detect_host_arch()

    if args.build_for_all_archs:
        if not SUPPORTED_ARCHS:
            print("\nWARNING: SUPPORTED_ARCHS (py/build.py) is empty; nothing to build.")
        for arch in SUPPORTED_ARCHS:
            build_engine(arch, build_type, jobs)
    else:
        if host_arch not in SUPPORTED_ARCHS:
            print(
                f"\nWARNING: this host's arch ('{host_arch}') is not in "
                "SUPPORTED_ARCHS (py/build.py). Please add it there and commit, "
                "so cloud tooling knows to build/distribute a bundle for it."
            )
        build_engine(host_arch, build_type, jobs)

    if os.path.isdir(arch_build_dir(host_arch)):
        link_host_arch_build(host_arch)
    else:
        print(
            f"\nNo build found for this host's arch ('{host_arch}') under "
            f"{arch_build_dir(host_arch)}; target/engine not updated. Run "
            "py/build.py without --build-for-all-archs to build it."
        )

    # 2. Install the front-end's npm dependencies so the engine can launch the
    #    Vite dev server (`npm run dev`) for human-vs-AI play.
    if not args.skip_web:
        if shutil.which("npm") is None:
            print(
                "\nWARNING: npm not found on PATH -- skipping web UI deps.\n"
                "Human-vs-AI web play needs Node.js/npm installed."
            )
        else:
            web_dir = os.path.join(ROOT, "web")
            if os.path.exists(os.path.join(web_dir, "package-lock.json")):
                run("npm ci --no-audit --no-fund", cwd=web_dir)
            else:
                run("npm install --no-audit --no-fund", cwd=web_dir)

    # 3. Clone + build the Macondo shell binary (only when first cloned).
    if not args.skip_macondo:
        if shutil.which("git") is None:
            print("\nWARNING: `git` not found on PATH -- skipping Macondo build.")
        elif shutil.which("go") is None:
            print("\nWARNING: `go` not found on PATH -- skipping Macondo build.")
        else:
            if not os.path.isdir(MACONDO_DIR):
                clone_and_build_macondo()
            elif not args.skip_macondo_tag_check:
                check_macondo_tag()

            # Make the installed lexica resolvable by the macondo subprocess.
            link_lexica_into_macondo()

    print_built_binaries(target_dir)

    print("\nBuild complete. Play a human-vs-AI game with:")
    print('    ./target/engine/play_game --player "--type=human" --player "--type=greedy"')


if __name__ == "__main__":
    main()
