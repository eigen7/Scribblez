"""Tests for the shared devenv_utils worktree logic and PR-flow teardown.

These exercise submodules/devenv_utils/worktrees.py (primary-checkout
resolution + enumeration) and pr_flow.py's local teardown helpers against a
throwaway git repo -- no Gitea, no Docker. The central case is the one the
merge incident got wrong: resolving the primary checkout from inside a feature
worktree must return the primary checkout, not the worktree.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, which is not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import pr_flow  # noqa: E402
from submodules.devenv_utils.worktrees import (  # noqa: E402
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
    # The crux of the incident fix: resolved from inside a feature worktree,
    # the primary is still the primary checkout, not the worktree itself.
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


def test_pr_head_branch():
    # While the branch exists, head.ref is the branch name.
    assert (
        pr_flow.pr_head_branch({"head": {"ref": "my-branch", "label": "my-branch"}}) == "my-branch"
    )
    # Once the branch is deleted (e.g. a web-UI merge), Gitea degrades head.ref
    # to refs/pull/N/head; head.label still carries the real name.
    assert (
        pr_flow.pr_head_branch({"head": {"ref": "refs/pull/9/head", "label": "my-branch"}})
        == "my-branch"
    )
    # A fork PR's label is owner-qualified; the owner prefix is stripped.
    assert (
        pr_flow.pr_head_branch({"head": {"ref": "refs/pull/9/head", "label": "fork:my-branch"}})
        == "my-branch"
    )


def test_teardown_branch_is_idempotent(repo: Path):
    add_worktree(repo, "feature", "feature")
    pr_flow.teardown_branch(repo, "feature", force=True)
    # Re-running against the now-absent branch and worktree must not raise --
    # this is what makes a re-run of `pr.py merge` after a partial failure safe.
    pr_flow.teardown_branch(repo, "feature", force=True)


def test_delete_local_branch_tolerates_missing(repo: Path):
    pr_flow.delete_local_branch(repo, "never-existed", force=True)


def test_delete_local_branch_reraises_real_errors(repo: Path):
    add_worktree(repo, "feature", "feature")
    # git refuses to delete a branch checked out in a live worktree. That is a
    # genuine failure, not the tolerated already-gone case, so it must surface
    # rather than be swallowed as idempotency.
    with pytest.raises(subprocess.CalledProcessError):
        pr_flow.delete_local_branch(repo, "feature", force=False)


def git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def init_repo_with_commit(root: Path, content: str):
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "f.txt").write_text(content)
    git(root, "add", "f.txt")
    git(root, "commit", "-q", "-m", content)


def test_sync_submodules_fetches_pointer_from_gitea(tmp_path: Path):
    """cmd_merge's submodule sync must pull a freshly referenced commit from
    gitea, not the submodule's upstream origin -- origin does not have it yet
    (its push is a later host-side step). Reproduces the exact merge failure:
    the recorded pointer lives only on gitea, absent from origin and from the
    superproject's own submodule clone.
    """
    allow = ["-c", "protocol.file.allow=always"]
    origin_sub = tmp_path / "origin_sub.git"
    gitea_sub = tmp_path / "gitea_sub.git"
    git(tmp_path, "init", "-q", "--bare", str(origin_sub))
    git(tmp_path, "init", "-q", "--bare", str(gitea_sub))

    # Submodule commit C1, published to origin; then C2, published only to gitea.
    seed = tmp_path / "seed"
    init_repo_with_commit(seed, "c1")
    git(seed, "remote", "add", "origin", str(origin_sub))
    git(seed, "push", "-q", "origin", "main")
    (seed / "f.txt").write_text("c2")
    git(seed, "add", "f.txt")
    git(seed, "commit", "-q", "-m", "c2")
    c2 = git_out(seed, "rev-parse", "HEAD")
    git(seed, "remote", "add", "gitea", str(gitea_sub))
    git(seed, "push", "-q", "gitea", "main")

    # Superproject with the submodule at C1, then bump the recorded pointer to
    # C2 without the superproject's clone ever obtaining C2.
    sup = tmp_path / "sup"
    init_repo_with_commit(sup, "root")
    git(sup, *allow, "submodule", "add", str(origin_sub), "submodules/dev")
    git(sup, "commit", "-q", "-m", "add submodule")
    sub = sup / "submodules" / "dev"
    git(sub, "remote", "add", "gitea", str(gitea_sub))
    git(sup, "update-index", "--cacheinfo", f"160000,{c2},submodules/dev")
    git(sup, "commit", "-q", "-m", "bump pointer to c2")

    assert not pr_flow.commit_present(sub, c2)  # missing before the sync
    pr_flow.sync_submodules(sup)
    # Working tree advanced to C2, sourced from gitea (origin never had it).
    assert git_out(sub, "rev-parse", "HEAD") == c2
