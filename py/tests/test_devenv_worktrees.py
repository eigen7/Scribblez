"""Tests for devenv_utils' worktree logic (worktrees.py: primary-checkout resolution
and enumeration) and pr_flow.py's local teardown helpers, against a throwaway git
repo -- no GitHub, no Docker.

The case that matters most: resolved from inside a feature worktree, the primary
checkout must still be the primary checkout, since every pr_flow command anchors
its git operations there.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# subtrees.* lives at the repo root, outside the py/-rooted import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from subtrees.devenv_utils import pr_flow  # noqa: E402
from subtrees.devenv_utils.worktrees import (  # noqa: E402
    primary_worktree,
    secondary_worktrees,
    worktree_for_branch,
)


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A primary checkout with a single commit on `main`."""
    root = tmp_path / "primary"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "file.txt").write_text("hello\n")
    git(root, "add", "file.txt")
    git(root, "commit", "-m", "initial")
    return root


def add_worktree(repo: Path, name: str, branch: str) -> Path:
    path = repo.parent / name
    git(repo, "worktree", "add", str(path), "-b", branch)
    return path


def test_primary_worktree_from_primary(repo: Path):
    assert primary_worktree(repo).resolve() == repo.resolve()


def test_primary_worktree_from_secondary(repo: Path):
    worktree = add_worktree(repo, "feature", "feature")
    assert primary_worktree(worktree).resolve() == repo.resolve()


def test_secondary_worktrees_excludes_primary(repo: Path):
    worktree = add_worktree(repo, "feature", "feature")
    secondary = secondary_worktrees(repo)
    assert [w.path.resolve() for w in secondary] == [worktree.resolve()]
    assert secondary[0].branch == "feature"
    # Same answer when anchored at the worktree rather than the primary.
    assert [w.path.resolve() for w in secondary_worktrees(worktree)] == [worktree.resolve()]


def test_worktree_for_branch(repo: Path):
    worktree = add_worktree(repo, "feature", "feature")
    found = worktree_for_branch(repo, "feature")
    assert found is not None and found.resolve() == worktree.resolve()
    assert worktree_for_branch(repo, "nonexistent") is None


def test_detached_head_worktree_has_no_branch(repo: Path):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    git(repo, "worktree", "add", "--detach", str(repo.parent / "detached"), head)
    secondary = secondary_worktrees(repo)
    assert len(secondary) == 1
    assert secondary[0].branch is None


def test_teardown_branch_removes_worktree_and_branch(repo: Path):
    worktree = add_worktree(repo, "feature", "feature")
    assert pr_flow.local_branch_exists(repo, "feature")
    assert pr_flow.teardown_branch(repo, "feature", force=True) is True
    assert worktree_for_branch(repo, "feature") is None
    assert not pr_flow.local_branch_exists(repo, "feature")
    assert not worktree.exists()


def test_teardown_branch_without_worktree_returns_false(repo: Path):
    # A branch that was never checked out in a worktree: teardown deletes it but
    # reports that no worktree was removed, so callers don't claim one was.
    git(repo, "branch", "loose")
    assert pr_flow.teardown_branch(repo, "loose", force=True) is False
    assert not pr_flow.local_branch_exists(repo, "loose")


def test_teardown_branch_is_idempotent(repo: Path):
    add_worktree(repo, "feature", "feature")
    pr_flow.teardown_branch(repo, "feature", force=True)
    # A second teardown of the absent branch and worktree must not raise: that
    # is what makes re-running `pr_flow.py cleanup` after a partial failure safe.
    pr_flow.teardown_branch(repo, "feature", force=True)


def test_delete_local_branch_tolerates_missing(repo: Path):
    pr_flow.delete_local_branch(repo, "never-existed", force=True)


def test_delete_local_branch_reraises_real_errors(repo: Path):
    add_worktree(repo, "feature", "feature")
    # git refuses to delete a branch checked out in a live worktree. That is a
    # real failure, not the tolerated already-gone case, so it must surface.
    with pytest.raises(subprocess.CalledProcessError):
        pr_flow.delete_local_branch(repo, "feature", force=False)
