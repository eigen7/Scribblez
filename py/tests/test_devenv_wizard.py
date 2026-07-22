"""Tests for SetupWizardTool's git-setup helpers (submodules/devenv_utils/
wizard.py) that don't need Docker or the Gitea service.

Only the local-config exclude helper is covered here, against a throwaway git
repo. The full wizard run is interactive and provisions services, so it is
validated by running it.
"""

import subprocess
import sys
from pathlib import Path

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils.wizard import SetupWizardTool  # noqa: E402


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    return root


def exclude_file(repo: Path) -> Path:
    return repo / ".git" / "info" / "exclude"


def test_exclude_local_config_adds_entry(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    SetupWizardTool._exclude_local_config(repo)
    assert "devenv.local.toml" in exclude_file(repo).read_text().splitlines()


def test_exclude_local_config_is_idempotent(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    SetupWizardTool._exclude_local_config(repo)
    SetupWizardTool._exclude_local_config(repo)
    lines = exclude_file(repo).read_text().splitlines()
    assert lines.count("devenv.local.toml") == 1


def test_exclude_local_config_preserves_existing_entries(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    exclude = exclude_file(repo)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("# git ls-files --others --exclude-from=.git/info/exclude\n*.log\n")
    SetupWizardTool._exclude_local_config(repo)

    lines = exclude.read_text().splitlines()
    assert "*.log" in lines
    assert "devenv.local.toml" in lines
