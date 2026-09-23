#!/usr/bin/env python3
"""Build Scribblez: the C++ engine, the web UI's npm packages, and Macondo.

The engine builds under target/archs/<arch>/ (this machine's CPU by default;
see --archs), and target/engine links to this machine's build. play_game
starts the web UI's dev server itself, so npm never needs running by hand.
The pinned Macondo checkout under the mount dir, whose data files the engine
reads, is cloned or moved to the pinned tag as needed.

Usage:
    py/build.py [--debug] [--clean] [-j N]

Then play a human-vs-AI game with:
    ./target/engine/play_game --player "--type=human" --player "--type=greedy"
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

from scribblez.hardware import default_thread_count
from setup_check import import_setup_common
from util.argparse_ext import ArgumentDefaultsHelpFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(ROOT, "target")
ARCHS_DIR = os.path.join(TARGET_DIR, "archs")

# Pinned Macondo release: cloned if absent, and an existing checkout is moved
# onto it (unless --skip-macondo-tag-sync). cloud/worker_deps.py fetches the
# same tag on workers.
MACONDO_TAG = "v0.13.2"
MOUNT_DIR = "/workspace/mount"
MACONDO_DIR = os.path.join(MOUNT_DIR, "macondo")
MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"

# setup_wizard.py downloads the .kwg lexica into <mount>/lexica/. Macondo looks
# for them by name under <macondo>/data/lexica/gaddag/, which
# link_lexica_into_macondo points at <mount>/lexica.
LEXICA_DIR = os.path.join(MOUNT_DIR, "lexica")
MACONDO_GADDAG_DIR = os.path.join(MACONDO_DIR, "data", "lexica", "gaddag")


def run_rc(cmd, cwd=None) -> int:
    """Like run(), but returns the exit code instead of exiting, for callers
    that must carry on after a failure (e.g. a parallel multi-arch build)."""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd or ROOT).returncode


def run(cmd, cwd=None):
    result_code = run_rc(cmd, cwd)
    if result_code:
        sys.exit(result_code)


def detect_host_arch() -> str:
    """This host's CPU microarchitecture as a GCC -march value (e.g.
    "alderlake", "znver3"). Asking the compiler what -march=native resolves to,
    rather than reading /proc/cpuinfo, guarantees a value it accepts."""
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


def arch_build_log(arch: str) -> str:
    return os.path.join(arch_build_dir(arch), "build.log")


# Strips ANSI escape sequences (colored diagnostics, cursor control) so build
# logs are plain text.
_STRIP_TTY_SED = r"sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g'"


def configure_engine(arch: str, build_type: str, *, live: bool, tests: bool) -> int:
    """Run the CMake configure step for one arch.

    `tests` enables the C++ test binaries, which only an arch this machine can
    execute may build (see SCRIBBLEZ_BUILD_TESTS in the top-level
    CMakeLists.txt). Without `live`, output goes to the arch's build.log
    instead of the terminal.
    """
    build_dir = arch_build_dir(arch)
    cmd = (
        f"cmake -S . -B {build_dir} -DCMAKE_BUILD_TYPE={build_type} "
        f"-DSCRIBBLEZ_MARCH={arch} -DSCRIBBLEZ_BUILD_TESTS={'ON' if tests else 'OFF'}"
    )
    if live:
        print(f"\nConfiguring arch '{arch}' ({build_type}) in {build_dir} ...")
        return run_rc(cmd)
    os.makedirs(build_dir, exist_ok=True)
    with open(arch_build_log(arch), "w") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        return subprocess.run(
            cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def build_engine(arch: str, build_type: str, jobs: int) -> int:
    """Configure and compile the engine, tests included, for this host's
    arch: the default build without --archs. Returns the first failing step's
    exit code, or 0.
    """
    build_dir = arch_build_dir(arch)
    rc = configure_engine(arch, build_type, live=True, tests=True)
    if rc:
        return rc
    print(f"\nCompiling arch '{arch}' (-j{jobs}) in {build_dir} ...")
    return run_rc(f"cmake --build {build_dir} -j{jobs}")


def build_all_archs(
    archs: list[str], build_type: str, total_jobs: int, host_arch: str
) -> list[str]:
    """Configure, then compile, every arch in `archs`. Returns the archs that
    failed.

    Configuring is cheap and not CPU-bound, so all archs configure
    concurrently. Compiling shares one pool of `total_jobs` job slots: each
    arch's build runs as a recursive submake under a single top-level
    `make -j{total_jobs}`, and GNU Make's jobserver hands an arch's slots to
    the others as soon as it finishes. A fixed per-arch split would leave
    cores idle once a fast or already-built arch finished. This depends on
    the Unix Makefiles generator; Ninja has no cross-process jobserver.

    Only one arch, the "primary" (`host_arch` if present, else the first),
    prints to the terminal. Every arch, the primary included, writes a
    plain-text target/archs/<arch>/build.log.
    """
    primary = host_arch if host_arch in archs else archs[0]

    # Start every log fresh. configure_engine(live=False) truncates the
    # non-primary logs, but the primary's live configure never touches its log,
    # and the compile step below appends.
    for arch in archs:
        log = arch_build_log(arch)
        if os.path.exists(log):
            os.remove(log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(archs)) as pool:
        configure_rcs = dict(
            zip(
                archs,
                pool.map(
                    lambda arch: configure_engine(
                        arch, build_type, live=(arch == primary), tests=(arch == host_arch)
                    ),
                    archs,
                ),
                strict=True,
            )
        )
    failed = [arch for arch, rc in configure_rcs.items() if rc]
    if failed:
        _print_log_locations(archs)
        return failed

    print(
        f"\nCompiling {len(archs)} arch(s) under a shared -j{total_jobs} job pool "
        f"(showing '{primary}' live) ..."
    )

    # Per-arch results can't be read back from the single top-level make, so a
    # failing recipe touches a marker file instead (cleared first, so a stale
    # one can't be misread). `|| touch <marker>` also makes every recipe
    # succeed as far as make is concerned, so one arch failing never stops the
    # others; no `-k` needed.
    fail_markers = {arch: os.path.join(arch_build_dir(arch), ".build_failed") for arch in archs}
    for marker in fail_markers.values():
        if os.path.exists(marker):
            os.remove(marker)

    # pipefail (via .SHELLFLAGS) makes a recipe's status $(MAKE)'s rather than
    # sed's or tee's, so the marker is touched on a real build failure. The
    # primary tees raw output to the terminal and a filtered copy to its log;
    # the others write only to their filtered logs.
    def recipe(arch: str) -> str:
        make_cmd = f"+$(MAKE) -C {arch_build_dir(arch)} 2>&1"
        log = arch_build_log(arch)
        if arch == primary:
            pipeline = f"{make_cmd} | tee >({_STRIP_TTY_SED} >> {log})"
        else:
            pipeline = f"{make_cmd} | {_STRIP_TTY_SED} >> {log}"
        return f"{arch}:\n\t{pipeline} || touch {fail_markers[arch]}\n\n"

    targets = " ".join(archs)
    rules = "".join(recipe(arch) for arch in archs)
    dispatch_makefile = (
        f"SHELL := /bin/bash\n.SHELLFLAGS := -o pipefail -c\n\n"
        f".PHONY: all {targets}\nall: {targets}\n\n{rules}"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mk", dir=TARGET_DIR, delete=False) as f:
        f.write(dispatch_makefile)
        dispatch_path = f.name
    try:
        dispatch_rc = run_rc(f"make -f {dispatch_path} -j{total_jobs} all")
    finally:
        os.unlink(dispatch_path)

    print(f"\nAbove output was for arch={primary}.")
    _print_log_locations(archs)

    failed = [arch for arch, marker in fail_markers.items() if os.path.exists(marker)]
    if dispatch_rc and not failed:
        # Recipes never fail as far as make knows, so a nonzero exit here is a
        # structural failure no marker captured (a scheduling or jobserver
        # error, a signal). Don't mistake the empty marker set for success.
        sys.exit(
            f"Build failed: the dispatch make exited {dispatch_rc} without any "
            "arch reporting a compile failure. See the output above."
        )
    return failed


def count_warnings(log_path: str) -> int:
    """Count compiler warning diagnostics (": warning: " lines) in a build
    log."""
    if not os.path.isfile(log_path):
        return 0
    with open(log_path, errors="replace") as f:
        return sum(1 for line in f if ": warning: " in line)


def _print_log_locations(archs: list[str]):
    print("\nTo see output for all archs, see:")
    for arch in archs:
        rel = os.path.relpath(arch_build_log(arch), ROOT)
        warnings = count_warnings(arch_build_log(arch))
        if warnings:
            plural = "" if warnings == 1 else "s"
            print(f"    {rel}  \033[33m<- {warnings} warning{plural}\033[0m")
        else:
            print(f"    {rel}")


def _replace_with_symlink(link_path: str, real_path: str):
    """Point `link_path` at `real_path` with a relative symlink, replacing
    whatever is there (a symlink, file or directory)."""
    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        shutil.rmtree(link_path)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(os.path.relpath(real_path, os.path.dirname(link_path)), link_path)


def link_host_arch_build(arch: str):
    """Point target/engine and target/compile_commands.json at this host
    arch's build, the fixed paths that tests, scripts, the FFI loader and
    clangd expect."""
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
    """Point Macondo's data/lexica/gaddag directory at <mount>/lexica, so
    Macondo's own shell resolves every installed lexicon, including ones added
    later. The symlink is relative, so it survives the mount dir moving.
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


def build_macondo_shell():
    os.makedirs(os.path.join(MACONDO_DIR, "bin"), exist_ok=True)
    run("go build -o bin/shell ./cmd/shell", cwd=MACONDO_DIR)


def clone_and_build_macondo():
    """Clone the pinned Macondo tag and build its shell binary. The shell is
    built only here and on a tag change, not on every run. If either step
    fails, the checkout is removed so the next run starts clean.
    """
    print(f"\nCloning Macondo {MACONDO_TAG} into {MACONDO_DIR} ...")
    try:
        run(
            f"git -c advice.detachedHead=false clone --branch {MACONDO_TAG} --depth 1 --quiet "
            f"{MACONDO_REPO_URL} {MACONDO_DIR}"
        )
        build_macondo_shell()
    except BaseException:
        shutil.rmtree(MACONDO_DIR, ignore_errors=True)
        raise


def macondo_git_output(*args) -> tuple[int, str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=MACONDO_DIR)
    return result.returncode, result.stdout.strip()


def current_macondo_tag() -> str:
    """The tag the Macondo checkout sits on, or "" if HEAD is not on an exact tag."""
    returncode, tag = macondo_git_output("describe", "--tags", "--exact-match", "HEAD")
    return tag if returncode == 0 else ""


def sync_macondo_tag():
    """Move an existing Macondo checkout onto MACONDO_TAG and rebuild its shell.

    The tag is fetched by name because the checkout is shallow and
    `clone --branch <tag>` leaves a refspec naming only that tag, so a plain
    `git fetch` would not see a newer one.

    Uncommitted edits to the Macondo source stop the build rather than being
    stashed or discarded: editing Macondo is a supported workflow (see
    --skip-macondo-tag-sync).
    """
    current = current_macondo_tag()
    if current == MACONDO_TAG:
        return
    _, dirty = macondo_git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print(
            f"\nError: macondo at {MACONDO_DIR} has uncommitted changes, so it "
            f"cannot be moved from '{current or '(no exact tag)'}' to '{MACONDO_TAG}'.\n"
            "Commit or stash them, or pass --skip-macondo-tag-sync to build "
            "against the checkout as it stands."
        )
        sys.exit(1)
    print(f"\nUpdating Macondo {current or '(no exact tag)'} -> {MACONDO_TAG} ...")
    run(
        f"git fetch --depth 1 --quiet origin refs/tags/{MACONDO_TAG}:refs/tags/{MACONDO_TAG}",
        cwd=MACONDO_DIR,
    )
    run(f"git -c advice.detachedHead=false checkout --quiet {MACONDO_TAG}", cwd=MACONDO_DIR)
    build_macondo_shell()


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
        "-a",
        "--archs",
        default="",
        help="comma-separated GCC -march values to build instead of this host's arch, each "
        "under target/archs/<arch>/ (tests are built only for this host's arch). Useful for "
        "pushing a bundle by hand for another CPU family; the dashboard builds the archs its "
        "machines need itself",
    )
    parser.add_argument(
        "--skip-web", action="store_true", help="skip installing the web UI npm dependencies"
    )
    parser.add_argument(
        "--skip-macondo",
        action="store_true",
        help="skip the Macondo steps: clone, tag sync, shell build and lexica link",
    )
    parser.add_argument(
        "--skip-macondo-tag-sync",
        action="store_true",
        help="leave the Macondo checkout on its current tag instead of moving it to "
        f"{MACONDO_TAG}; for when you are editing the Macondo source",
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
        # A CMake build directly under target/, not under target/archs/<arch>/:
        # its cache paths can't be reused, so reconfigure from scratch.
        print(f"\nFound a pre-multi-arch build at {target_dir}; removing it to reconfigure.")
        shutil.rmtree(target_dir)

    # 1. The C++ engine, once per requested arch.
    build_type = "Debug" if args.debug else "Release"
    jobs = args.jobs or default_thread_count()
    host_arch = detect_host_arch()

    if args.archs:
        archs = sorted({a.strip() for a in args.archs.split(",") if a.strip()})
        failed = build_all_archs(archs, build_type, jobs, host_arch)
        if failed:
            sys.exit(f"Build failed for arch(s): {', '.join(sorted(failed))}")
    else:
        rc = build_engine(host_arch, build_type, jobs)
        if rc:
            sys.exit(rc)

    if os.path.isdir(arch_build_dir(host_arch)):
        link_host_arch_build(host_arch)
    else:
        print(
            f"\nNo build found for this host's arch ('{host_arch}') under "
            f"{arch_build_dir(host_arch)}; target/engine not updated. Run "
            "py/build.py without --archs to build it."
        )

    # 2. The web UI's npm packages, so play_game can start the Vite dev server.
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

    # 3. Macondo: clone, or move to MACONDO_TAG; the shell is rebuilt only then.
    if not args.skip_macondo:
        if shutil.which("git") is None:
            print("\nWARNING: `git` not found on PATH -- skipping Macondo build.")
        elif shutil.which("go") is None:
            print("\nWARNING: `go` not found on PATH -- skipping Macondo build.")
        else:
            if not os.path.isdir(MACONDO_DIR):
                clone_and_build_macondo()
            elif not args.skip_macondo_tag_sync:
                sync_macondo_tag()

            link_lexica_into_macondo()

    print_built_binaries(target_dir)

    print("\nBuild complete. Play a human-vs-AI game with:")
    print('    ./target/engine/play_game --player "--type=human" --player "--type=greedy"')


if __name__ == "__main__":
    main()
