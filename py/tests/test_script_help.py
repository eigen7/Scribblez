"""Every argparse-driven script and tool must at least print its --help.

--help imports the script and formats every help string, so it catches stale
imports and help text argparse cannot format (an unescaped `%`).
"""

import subprocess
import sys
from pathlib import Path

import pytest

PY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    p
    for p in [*PY_DIR.glob("scripts/**/*.py"), *PY_DIR.glob("tools/*.py")]
    if "argparse" in p.read_text()
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(PY_DIR)))
def test_help_runs(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
