"""Tests for the host-side publish helpers (publish.py) and the pre-push guard.

The pure/logic pieces are exercised against throwaway git repos -- no GitHub, no
Gitea, no host. The full `git publish` (which pushes to GitHub) is validated by
running it on the host.
"""

import subprocess
import sys
from pathlib import Path

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import prepush_guard, publish  # noqa: E402


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def init_repo(root: Path, content: str = "x"):
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "T")
    git(root, "config", "user.email", "t@t")
    (root / "f").write_text(content)
    git(root, "add", "f")
    git(root, "commit", "-q", "-m", content)


def test_is_origin_push():
    assert prepush_guard.is_origin_push("origin", "https://github.com/x/y.git")
    assert prepush_guard.is_origin_push("whatever", "git@github.com:x/y.git")
    assert not prepush_guard.is_origin_push("gitea", "http://localhost:3001/dshin/y.git")


def test_gitea_read_url_strips_creds_and_derives_submodule(tmp_path: Path):
    # A bare origin for the submodule, named so its basename is `devenv_utils`.
    sub_origin = tmp_path / "devenv_utils.git"
    git(tmp_path, "init", "-q", "--bare", str(sub_origin))
    seed = tmp_path / "seed"
    init_repo(seed)
    git(seed, "remote", "add", "origin", str(sub_origin))
    git(seed, "push", "-q", "origin", "main")

    sup = tmp_path / "sup"
    init_repo(sup)
    git(sup, "remote", "add", "gitea", "http://claude:pw@localhost:3001/dshin/scribblez.git")
    git(sup, "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), "sub")
    git(sup, "commit", "-q", "-m", "add sub")

    # Parent: credentials dropped, backend :3001 -> web :3000.
    assert publish.gitea_read_url(sup) == "http://localhost:3000/dshin/scribblez.git"
    # Submodule: same host/owner, repo name from the submodule's origin basename,
    # needing no `gitea` remote of its own.
    assert publish.gitea_read_url(sup, "sub") == "http://localhost:3000/dshin/devenv_utils.git"


def test_teardown_removes_only_merged_worktrees(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)

    # A merged worktree: its branch tip is an ancestor of main (no new commits).
    git(repo, "worktree", "add", str(tmp_path / "merged"), "-b", "merged")
    # A live worktree: a commit main doesn't have, so not merged.
    git(repo, "worktree", "add", str(tmp_path / "live"), "-b", "live")
    (tmp_path / "live" / "g").write_text("z")
    git(tmp_path / "live", "add", "g")
    git(tmp_path / "live", "commit", "-q", "-m", "z")

    publish.teardown_merged_worktrees(repo)

    branches = git_out(repo, "branch", "--format=%(refname:short)").split()
    assert "merged" not in branches and not (tmp_path / "merged").exists()
    assert "live" in branches and (tmp_path / "live").exists()


def test_is_ancestor(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo, "one")
    first = git_out(repo, "rev-parse", "HEAD")
    (repo / "f").write_text("two")
    git(repo, "add", "f")
    git(repo, "commit", "-q", "-m", "two")
    assert publish.is_ancestor(repo, first, "HEAD")
    assert not publish.is_ancestor(repo, "HEAD", first)
