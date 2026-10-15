"""Every argparse-driven script and tool must at least print its --help.

--help imports the script and formats every help string, so it catches stale
imports and help text argparse cannot format (an unescaped `%`).
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    p
    for p in [*PY_DIR.glob("scripts/**/*.py"), *PY_DIR.glob("tools/*.py")]
    if "argparse" in p.read_text()
)

# Each run is mostly a fresh interpreter importing torch (~1.5 s of one core),
# so they run side by side; capped so the suite leaves most of the machine to
# whatever else is running.
CONCURRENCY = 8


def _run_help(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120
    )


@pytest.fixture(scope="module")
def help_runs(request):
    """The selected scripts' --help runs, all submitted up front; each test
    waits on its own."""
    selected = [
        item.callspec.params["script"]
        for item in request.session.items
        if getattr(item, "function", None) is test_help_runs
    ]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        yield {script: pool.submit(_run_help, script) for script in selected}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(PY_DIR)))
def test_help_runs(script, help_runs):
    result = help_runs[script].result()
    assert result.returncode == 0, result.stderr
