#!/usr/bin/env python3
"""Build Scribblez: compile the C++ engine and install the web UI's npm deps.

This is the one command you run before playing. The build step installs the
front-end's npm packages, and the engine (play_game) launches the Vite dev
server itself at play time -- so you never invoke npm by hand.

Usage:
    ./build.py [--debug] [--clean] [-j N]

Then play a human-vs-AI game with:
    ./build/engine/play_game --player "--type=human" --player "--type=greedy"
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true",
                        help="debug build (default: Release)")
    parser.add_argument("--clean", action="store_true",
                        help="remove the build/ directory before configuring")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="parallel build jobs (default: all CPUs)")
    parser.add_argument("--skip-web", action="store_true",
                        help="skip installing the web UI npm dependencies")
    args = parser.parse_args()

    build_dir = os.path.join(ROOT, "build")
    if args.clean and os.path.isdir(build_dir):
        shutil.rmtree(build_dir)

    # 1. Configure + compile the C++ engine.
    build_type = "Debug" if args.debug else "Release"
    run(f"cmake -S . -B build -DCMAKE_BUILD_TYPE={build_type}")
    jobs = args.jobs or (os.cpu_count() or 1)
    run(f"cmake --build build -j{jobs}")

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

    print("\nBuild complete. Play a human-vs-AI game with:")
    print('    ./build/engine/play_game --player "--type=human" --player "--type=greedy"')


if __name__ == "__main__":
    main()
