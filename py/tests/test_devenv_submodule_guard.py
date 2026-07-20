"""Tests for the submodule pointer/sync hooks
(submodules/devenv_utils/submodule_guard.py).

The hook actions are exercised as plain functions against throwaway git
repos: a superproject holding a real submodule, with the submodule's source
repo standing in for its upstream. Hook installation itself is the wizard's
job and isn't covered here.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import submodule_guard  # noqa: E402


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def init_repo(root: Path, content: str = "x") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "T")
    git(root, "config", "user.email", "t@t")
    (root / "f").write_text(content)
    git(root, "add", "f")
    git(root, "commit", "-q", "-m", content)
    return root


def commit(repo: Path, content: str):
    (repo / "f").write_text(content)
    git(repo, "add", "f")
    git(repo, "commit", "-q", "-m", content)


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, str, str]:
    """A superproject whose submodule `sub` is pinned at commit A, while the
    submodule's upstream also has a later commit B. Returns (super, A, B)."""
    upstream = init_repo(tmp_path / "upstream", "A")
    commit(upstream, "B")
    a_sha = git_out(upstream, "rev-parse", "main~1")
    b_sha = git_out(upstream, "rev-parse", "main")

    super_ = init_repo(tmp_path / "super", "super")
    git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(upstream), "sub")
    git(super_ / "sub", "checkout", "-q", a_sha)
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin A")
    return super_, a_sha, b_sha


def stage_pointer(super_: Path, sha: str):
    git(super_ / "sub", "checkout", "-q", sha)
    git(super_, "add", "sub")


def sub_head(super_: Path) -> str:
    return git_out(super_ / "sub", "rev-parse", "HEAD")


# ---- pre-commit: the backward-pointer guard -----------------------------


def test_check_allows_clean_index(repos):
    super_, _, _ = repos
    submodule_guard.check(super_)


def test_check_allows_forward_bump(repos):
    super_, _, b_sha = repos
    stage_pointer(super_, b_sha)
    submodule_guard.check(super_)


def test_check_blocks_rewind(repos):
    super_, a_sha, b_sha = repos
    stage_pointer(super_, b_sha)
    git(super_, "commit", "-q", "-m", "bump to B")
    stage_pointer(super_, a_sha)
    with pytest.raises(SystemExit, match="backward"):
        submodule_guard.check(super_)


def test_check_allows_unrelated_changes(repos):
    super_, _, _ = repos
    (super_ / "g").write_text("g")
    git(super_, "add", "g")
    submodule_guard.check(super_)


def test_check_allows_new_submodule(repos, tmp_path: Path):
    super_, _, _ = repos
    other = init_repo(tmp_path / "other-upstream", "o")
    git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(other), "sub2")
    submodule_guard.check(super_)


# ---- sync: re-syncing stale checkouts -----------------------------------


def make_stale(repos):
    """Record pointer B while the checkout sits at A -- the state a rebase
    over a pointer bump leaves behind."""
    super_, a_sha, b_sha = repos
    stage_pointer(super_, b_sha)
    git(super_, "commit", "-q", "-m", "bump to B")
    git(super_ / "sub", "checkout", "-q", a_sha)


def test_sync_updates_stale_checkout(repos, capsys):
    super_, _, b_sha = repos
    make_stale(repos)
    submodule_guard.sync(super_)
    assert sub_head(super_) == b_sha
    assert "synced sub" in capsys.readouterr().out


def test_sync_noop_when_in_sync(repos, capsys):
    super_, a_sha, _ = repos
    submodule_guard.sync(super_)
    assert sub_head(super_) == a_sha
    assert capsys.readouterr().out == ""


def test_sync_skips_dirty_checkout(repos, capsys):
    super_, a_sha, _ = repos
    make_stale(repos)
    (super_ / "sub" / "f").write_text("edited")
    submodule_guard.sync(super_)
    assert sub_head(super_) == a_sha
    assert (super_ / "sub" / "f").read_text() == "edited"
    assert "uncommitted changes" in capsys.readouterr().err


def test_sync_leaves_checkout_ahead_of_pointer(repos, capsys):
    # The pointer records A while the checkout sits at the later B: syncing
    # would rewind past commits the pointer lacks, so it must not happen.
    super_, _, b_sha = repos
    git(super_ / "sub", "checkout", "-q", b_sha)
    submodule_guard.sync(super_)
    assert sub_head(super_) == b_sha
    assert "commits the recorded pointer lacks" in capsys.readouterr().err


def test_sync_leaves_unpopulated_submodule(repos):
    super_, _, _ = repos
    make_stale(repos)
    git(super_, "submodule", "deinit", "-f", "sub")
    submodule_guard.sync(super_)
    assert not (super_ / "sub" / "f").exists()
