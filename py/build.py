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

# CPU microarchitectures (GCC/Clang -march values) this project builds
# bundles for, e.g. for cloud tooling to upload one binary bundle per arch and
# have each worker fetch the one matching its CPU. Grows by hand: when
# building on a host whose arch isn't listed here, build_engine() below warns
# so the operator can add it and commit.
SUPPORTED_ARCHS = ["alderlake", "tigerlake", "x86-64", "znver2", "znver4"]

# Pinned Macondo release. build.py will clone this tag if the repo is absent,
# and will move an existing checkout onto it if it is at a different tag
# (unless --skip-macondo-tag-sync is passed).
MACONDO_TAG = "v0.13.2"
MOUNT_DIR = "/workspace/mount"
MACONDO_DIR = os.path.join(MOUNT_DIR, "macondo")
MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"

# Lexicon .kwg files are downloaded into <mount>/lexica/ by setup_wizard.py.
# Macondo expects to find them under <macondo>/data/lexica/gaddag/, keyed by
# lexicon name; we point that directory at <mount>/lexica so the macondo
# subprocess can resolve them.
LEXICA_DIR = os.path.join(MOUNT_DIR, "lexica")
MACONDO_GADDAG_DIR = os.path.join(MACONDO_DIR, "data", "lexica", "gaddag")


def run_rc(cmd, cwd=None) -> int:
    """Like run(), but returns the exit code instead of exiting the process --
    for callers that need to keep going after a failure (e.g. a parallel arch
    build, where one arch failing shouldn't kill its still-running siblings)."""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd or ROOT).returncode


def run(cmd, cwd=None):
    result_code = run_rc(cmd, cwd)
    if result_code:
        sys.exit(result_code)


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


def arch_build_log(arch: str) -> str:
    return os.path.join(arch_build_dir(arch), "build.log")


# Strips ANSI/terminal escape sequences (colored diagnostics, cursor control)
# so a log file reads as plain text regardless of what the tool that produced
# it thought it was writing to.
_STRIP_TTY_SED = r"sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g'"


def configure_engine(arch: str, build_type: str, *, live: bool, tests: bool) -> int:
    """Run just the CMake configure step for one CPU microarchitecture.

    `tests` builds the C++ test binaries, which only an arch this machine can
    execute may do (see SCRIBBLEZ_BUILD_TESTS in the top-level CMakeLists.txt).

    When `live` is False, output is redirected to the arch's build.log
    instead of the terminal, for the non-primary archs of a multi-arch build
    (see build_all_archs).
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
    """Configure + compile the C++ engine for this host's own arch, tests
    included (the default single-arch build; build_all_archs handles the
    multi-arch case).

    Returns the exit code of the first failing step, or 0.
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
    """Configure then build every arch in `archs`.

    Configuring (`cmake -S -B`) is comparatively cheap and not CPU-bound, so
    every arch configures concurrently. Compiling is CPU-bound; statically
    splitting total_jobs across archs up front would leave cores idle once a
    faster (or already-cached) arch finishes early. Instead, every arch's
    build is driven as a recursive submake ("+$(MAKE) -C <dir>") under one
    top-level `make -j{total_jobs}`. GNU Make's jobserver protocol then pools
    job tokens across all of them: an arch that finishes frees its tokens
    immediately for the arch(s) still compiling, keeping the machine
    saturated without exceeding total_jobs concurrent compiler invocations,
    regardless of how unevenly the archs finish. This relies on the project's
    Unix Makefiles generator (CMAKE_MAKE_PROGRAM is gmake here); it would need
    rework under Ninja, which has no equivalent cross-process job-sharing
    protocol.

    Building N archs at once means N streams of compiler output racing for
    the same terminal. Rather than interleaving (or grouping) all of them,
    only one arch -- the "primary", `host_arch` if it's in `archs` else the
    first entry -- prints live; every other arch's output goes straight to
    its own target/archs/<arch>/build.log with nothing printed live. Every
    arch's log is written regardless (including the primary's), with
    terminal escape sequences stripped so the files stay plain text even
    though the primary's live view keeps them.

    Returns the archs whose configure or build step failed (empty on
    success).
    """
    primary = host_arch if host_arch in archs else archs[0]

    # Every arch's log gets rebuilt from scratch this run. Non-primary archs'
    # logs get recreated by configure_engine(live=False) below regardless,
    # but the primary's doesn't touch its log at all during the (live)
    # configure step, so without this a repeat run would silently append
    # this run's build output after last run's stale leftovers.
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

    # A recipe's own exit code can't be read back from the single `make`
    # subprocess we invoke below, so each one records a real failure via a
    # marker file instead (cleared up front so a stale marker from a prior
    # run can't be misread as this run failing). `|| touch <marker>` also
    # makes every recipe report success to `make` itself, so one arch failing
    # never stops the others from being scheduled -- there's no need for `-k`.
    fail_markers = {arch: os.path.join(arch_build_dir(arch), ".build_failed") for arch in archs}
    for marker in fail_markers.values():
        if os.path.exists(marker):
            os.remove(marker)

    # pipefail (via .SHELLFLAGS) makes each recipe's exit status reflect
    # $(MAKE) rather than tee/sed (which almost never fail), so
    # `|| touch <marker>` actually triggers on a real build failure. The
    # primary's recipe tees its raw (uncleaned) output live to the terminal
    # while also piping a filtered copy into its log via process
    # substitution; every other arch pipes straight into a filtered log with
    # nothing going to the terminal at all.
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
        # Every recipe swallows its own failure with `|| touch <marker>` and so
        # reports success to make, which means a nonzero exit from the dispatch
        # make can only be a structural failure no per-arch marker captured
        # (make couldn't schedule a recipe, a jobserver error, a signal). Fail
        # the build rather than mistaking that empty marker set for success.
        sys.exit(
            f"Build failed: the dispatch make exited {dispatch_rc} without any "
            "arch reporting a compile failure. See the output above."
        )
    return failed


def count_warnings(log_path: str) -> int:
    """Count compiler warning diagnostics in a build log.

    GCC and Clang both emit "warning:" in every warning diagnostic line
    (e.g. "foo.cpp:12:3: warning: unused variable [-Wunused-variable]").
    """
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


def build_macondo_shell():
    os.makedirs(os.path.join(MACONDO_DIR, "bin"), exist_ok=True)
    run("go build -o bin/shell ./cmd/shell", cwd=MACONDO_DIR)


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

    The checkout is shallow, and `clone --branch <tag>` leaves behind a fetch
    refspec naming only that one tag, so a plain `git fetch` would not see any
    newer tag -- the wanted one has to be fetched by name.

    Local edits to the Macondo source are left alone: they are a supported way
    to work (see --skip-macondo-tag-sync), and silently stashing or discarding
    them to move tags would be worse than stopping.
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
        "--skip-macondo-tag-sync",
        action="store_true",
        help="leave the macondo checkout on whatever tag it is on instead of "
        f"moving it to the expected one ({MACONDO_TAG}); useful when "
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
        else:
            failed = build_all_archs(SUPPORTED_ARCHS, build_type, jobs, host_arch)
            if failed:
                sys.exit(f"Build failed for arch(s): {', '.join(sorted(failed))}")
    else:
        if host_arch not in SUPPORTED_ARCHS:
            print(
                f"\nWARNING: this host's arch ('{host_arch}') is not in "
                "SUPPORTED_ARCHS (py/build.py). Please add it there and commit, "
                "so cloud tooling knows to build/distribute a bundle for it."
            )
        rc = build_engine(host_arch, build_type, jobs)
        if rc:
            sys.exit(rc)

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

    # 3. Clone + build the Macondo shell binary (only when first cloned, or
    #    when MACONDO_TAG has moved since).
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

            # Make the installed lexica resolvable by the macondo subprocess.
            link_lexica_into_macondo()

    print_built_binaries(target_dir)

    print("\nBuild complete. Play a human-vs-AI game with:")
    print('    ./target/engine/play_game --player "--type=human" --player "--type=greedy"')


if __name__ == "__main__":
    main()
