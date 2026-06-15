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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pinned Macondo release. build.py will clone this tag if the repo is absent,
# and will error if the existing checkout is at a different tag (unless
# --skip-macondo-tag-check is passed).
MACONDO_TAG = "v0.13.1"
MOUNT_DIR = "/workspace/mount"
MACONDO_DIR = os.path.join(MOUNT_DIR, "macondo")
MACONDO_REPO_URL = "https://github.com/domino14/macondo.git"

# Lexicon .kwg files are downloaded into <mount>/lexica/ by setup_wizard.py.
# Macondo expects to find them under <macondo>/data/lexica/gaddag/, keyed by
# lexicon name; we symlink them there after cloning so the macondo subprocess
# can resolve them.
LEXICA_DIR = os.path.join(MOUNT_DIR, "lexica")
MACONDO_GADDAG_DIR = os.path.join(MACONDO_DIR, "data", "lexica", "gaddag")


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def link_lexica_into_macondo():
    """Symlink <mount>/lexica/*.kwg into Macondo's data/lexica/gaddag/ dir.

    Macondo looks up its kwg by lexicon name from there. Uses relative symlinks
    so they survive the mount dir being moved. Safe to run repeatedly.
    """
    if not os.path.isdir(LEXICA_DIR):
        print(f"\nNo lexica dir at {LEXICA_DIR}; skipping macondo lexica links.\n"
              "Run ./setup_wizard.py to install lexica.")
        return
    kwgs = sorted(f for f in os.listdir(LEXICA_DIR) if f.endswith(".kwg"))
    if not kwgs:
        print(f"\nNo .kwg files in {LEXICA_DIR}; skipping macondo lexica links.")
        return
    os.makedirs(MACONDO_GADDAG_DIR, exist_ok=True)
    for name in kwgs:
        link = os.path.join(MACONDO_GADDAG_DIR, name)
        target = os.path.join(LEXICA_DIR, name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(os.path.relpath(target, MACONDO_GADDAG_DIR), link)
    print(f"\nLinked {len(kwgs)} lexica into {MACONDO_GADDAG_DIR}: "
          f"{', '.join(os.path.splitext(k)[0] for k in kwgs)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true",
                        help="debug build (default: Release)")
    parser.add_argument("--clean", action="store_true",
                        help="remove the target/ directory before configuring")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="parallel build jobs (default: all CPUs)")
    parser.add_argument("--skip-web", action="store_true",
                        help="skip installing the web UI npm dependencies")
    parser.add_argument("--skip-macondo", action="store_true",
                        help="skip rebuilding the Macondo shell binary")
    parser.add_argument("--skip-macondo-tag-check", action="store_true",
                        help="skip verifying that the macondo checkout is at "
                             f"the expected tag ({MACONDO_TAG}); useful when "
                             "fiddling with the macondo source")
    args = parser.parse_args()

    target_dir = os.path.join(ROOT, "target")
    if args.clean and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    # 1. Configure + compile the C++ engine.
    build_type = "Debug" if args.debug else "Release"
    run(f"cmake -S . -B target -DCMAKE_BUILD_TYPE={build_type}")
    jobs = args.jobs or (os.cpu_count() or 1)
    run(f"cmake --build target -j{jobs}")

    # 2. Install the front-end's npm dependencies so the engine can launch the
    #    Vite dev server (`npm run dev`) for human-vs-AI play.
    if not args.skip_web:
        if shutil.which("npm") is None:
            print("\nWARNING: npm not found on PATH -- skipping web UI deps.\n"
                  "Human-vs-AI web play needs Node.js/npm installed.")
        else:
            web_dir = os.path.join(ROOT, "web")
            if os.path.exists(os.path.join(web_dir, "package-lock.json")):
                run("npm ci --no-audit --no-fund", cwd=web_dir)
            else:
                run("npm install --no-audit --no-fund", cwd=web_dir)

    # 3. Clone (if absent) and build the Macondo shell binary.
    if not args.skip_macondo:
        if shutil.which("git") is None:
            print("\nWARNING: `git` not found on PATH -- skipping Macondo build.")
        elif shutil.which("go") is None:
            print("\nWARNING: `go` not found on PATH -- skipping Macondo build.")
        else:
            if not os.path.isdir(MACONDO_DIR):
                print(f"\nCloning Macondo {MACONDO_TAG} into {MACONDO_DIR} ...")
                run(f"git -c advice.detachedHead=false clone --branch {MACONDO_TAG} --depth 1 --quiet "
                    f"{MACONDO_REPO_URL} {MACONDO_DIR}")
            elif not args.skip_macondo_tag_check:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--exact-match", "HEAD"],
                    capture_output=True, text=True, cwd=MACONDO_DIR,
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
            os.makedirs(os.path.join(MACONDO_DIR, "bin"), exist_ok=True)
            run("go build -o bin/shell ./cmd/shell", cwd=MACONDO_DIR)

            # Make the installed lexica resolvable by the macondo subprocess.
            link_lexica_into_macondo()

    print("\nBuild complete. Play a human-vs-AI game with:")
    print('    ./target/engine/play_game --player "--type=human" --player "--type=greedy"')


if __name__ == "__main__":
    main()
