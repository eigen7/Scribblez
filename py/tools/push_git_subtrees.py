#!/usr/bin/env python3
"""Push each git subtree under subtrees/ to its upstream branch."""
import sys
from pathlib import Path

# Put py/ on sys.path so `setup_check` resolves when this script is run
# directly from outside the dev container (inside, scribblez.pth handles it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setup_check import import_setup_common

if __name__ == "__main__":
    import_setup_common().dev_tool().push_git_subtrees_cli("subtrees")
