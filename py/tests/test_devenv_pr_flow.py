"""Tests for pr_flow.py's pure decision helpers.

Only the ancestry / zero-commit logic that gates whether a PR is opened is
covered here, against throwaway git repos. The Gitea-facing subcommands
(worktree/create/abandon) need the live service and are validated by running
them.
"""

import subprocess
import sys
from pathlib import Path

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import pr_flow  # noqa: E402


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "T")
    git(root, "config", "user.email", "t@t")
    (root / "f").write_text("x")
    git(root, "add", "f")
    git(root, "commit", "-q", "-m", "init")
    return root


def commit(repo: Path, name: str):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)


def test_is_ancestor(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    first = git_out(repo, "rev-parse", "HEAD")
    commit(repo, "two")
    assert pr_flow.is_ancestor(repo, first, "HEAD")
    assert not pr_flow.is_ancestor(repo, "HEAD", first)


def test_is_ancestor_unknown_commit_is_not_ancestor(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    assert not pr_flow.is_ancestor(repo, "0" * 40, "HEAD")


def test_branch_adds_commits_true_when_branch_ahead(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, "work")
    assert pr_flow.branch_adds_commits(repo, "feature")


def test_branch_adds_commits_false_for_submodule_only_branch(tmp_path: Path):
    # A branch cut from main with no superproject commits -- the shape a
    # submodule-only change leaves -- adds nothing over main.
    repo = init_repo(tmp_path / "repo")
    git(repo, "checkout", "-q", "-b", "feature")
    assert not pr_flow.branch_adds_commits(repo, "feature")


def test_branch_adds_commits_false_when_branch_behind(tmp_path: Path):
    repo = init_repo(tmp_path / "repo")
    git(repo, "branch", "stale")  # points at the initial commit
    commit(repo, "advance-main")
    assert not pr_flow.branch_adds_commits(repo, "stale")
