"""Tests for the main-lockstep hooks (submodules/devenv_utils/commit_guard.py).

The hook actions are exercised as plain functions against throwaway git
repos, with a bare repo standing in for the Gitea remote -- no server, no
Docker. Hook installation itself is the wizard's job and isn't covered here.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import commit_guard  # noqa: E402


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


def bare_main(anchor: Path, bare: Path) -> str:
    """The bare repo's main tip, read via ls-remote: agent harnesses set
    safe.bareRepository=explicit, which forbids running git *inside* a bare
    repo but not addressing it as a remote."""
    return git_out(anchor, "ls-remote", str(bare), "main").split()[0]


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    """A repo whose `gitea` remote is a bare repo holding the same main."""
    repo = init_repo(tmp_path / "repo")
    bare = tmp_path / "gitea.git"
    git(tmp_path, "init", "-q", "--bare", str(bare))
    git(repo, "remote", "add", "gitea", str(bare))
    git(repo, "push", "-q", "gitea", "main")
    return repo, bare


def advance_gitea(tmp_path: Path, bare: Path):
    """Land a commit on the bare repo's main that the test repo doesn't have --
    the state a browser-merged PR leaves behind."""
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(bare), str(other))
    git(other, "config", "user.name", "T")
    git(other, "config", "user.email", "t@t")
    (other / "merged").write_text("merged")
    git(other, "add", "merged")
    git(other, "commit", "-q", "-m", "merged on gitea")
    git(other, "push", "-q", "origin", "main")


def test_check_allows_when_synced(pair):
    repo, _ = pair
    commit_guard.check(repo)


def test_check_allows_when_local_ahead(pair):
    repo, _ = pair
    commit(repo, "local-only")
    commit_guard.check(repo)


def test_check_blocks_unpublished_merges(pair, tmp_path: Path):
    repo, bare = pair
    advance_gitea(tmp_path, bare)
    with pytest.raises(SystemExit, match="git publish"):
        commit_guard.check(repo)


def test_check_ignores_other_branches(pair, tmp_path: Path):
    repo, bare = pair
    advance_gitea(tmp_path, bare)
    git(repo, "checkout", "-q", "-b", "feature")
    commit_guard.check(repo)


def test_check_allows_when_gitea_unreachable(pair, tmp_path: Path, capsys):
    repo, _ = pair
    git(repo, "remote", "set-url", "gitea", str(tmp_path / "gone.git"))
    commit_guard.check(repo)
    assert "could not reach Gitea" in capsys.readouterr().err


def test_sync_mirrors_tip(pair, capsys):
    repo, bare = pair
    commit(repo, "local")
    commit_guard.sync(repo)
    assert bare_main(repo, bare) == git_out(repo, "rev-parse", "main")
    assert "mirrored main -> gitea" in capsys.readouterr().out


def test_sync_reports_nonff_without_failing(pair, tmp_path: Path, capsys):
    repo, bare = pair
    advance_gitea(tmp_path, bare)
    gitea_tip = bare_main(repo, bare)
    commit(repo, "local")
    # The push is rejected (non-fast-forward), but a post-commit hook cannot
    # un-commit: sync must return normally, leave Gitea untouched, and print
    # the reconciliation path.
    commit_guard.sync(repo)
    assert bare_main(repo, bare) == gitea_tip
    assert "git publish" in capsys.readouterr().err


def test_sync_ignores_other_branches(pair):
    repo, bare = pair
    gitea_tip = bare_main(repo, bare)
    git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, "feature work")
    commit_guard.sync(repo)
    assert bare_main(repo, bare) == gitea_tip


def test_sync_mirrors_with_unpublished_submodule_commit(pair, tmp_path: Path):
    # Reproduce the host failure: main records a submodule commit that exists in
    # the submodule clone only as a local object (in no remote-tracking ref),
    # with push.recurseSubmodules=check set as the wizard installs it. The
    # same-machine mirror must still push, via --recurse-submodules=no.
    repo, bare = pair
    sub_upstream = init_repo(tmp_path / "sub_upstream", "A")
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(sub_upstream), "sub")
    git(repo, "commit", "-q", "-m", "add sub")
    # A submodule commit B that never reached the submodule's remote.
    commit(repo / "sub", "B")
    git(repo, "add", "sub")
    git(repo, "commit", "-q", "-m", "bump sub to B")
    git(repo, "config", "push.recurseSubmodules", "check")

    commit_guard.sync(repo)

    assert bare_main(repo, bare) == git_out(repo, "rev-parse", "main")
