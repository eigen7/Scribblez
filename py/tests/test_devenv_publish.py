"""Tests for the host-side publish helpers (publish.py) and the pre-push guard.

The pure/logic pieces are exercised against throwaway git repos -- no GitHub, no
Gitea, no host. The full `git publish` (which pushes to GitHub) is validated by
running it on the host.
"""

import subprocess
import sys
from pathlib import Path

import pytest

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
    assert not prepush_guard.is_origin_push("gitea", "http://localhost:3000/dshin/y.git")


def test_gitea_read_url_derives_submodule_from_parent_remote(tmp_path: Path):
    # A bare origin for the submodule, named so its basename is `devenv_utils`.
    sub_origin = tmp_path / "devenv_utils.git"
    git(tmp_path, "init", "-q", "--bare", str(sub_origin))
    seed = tmp_path / "seed"
    init_repo(seed)
    git(seed, "remote", "add", "origin", str(sub_origin))
    git(seed, "push", "-q", "origin", "main")

    sup = tmp_path / "sup"
    init_repo(sup)
    # The canonical credential-free web-port remote URL (gitea_client.py).
    git(sup, "remote", "add", "gitea", "http://localhost:3000/dshin/scribblez.git")
    git(sup, "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), "sub")
    git(sup, "commit", "-q", "-m", "add sub")

    # Parent: the canonical remote URL fetches from the host as-is.
    assert publish.gitea_read_url(sup) == "http://localhost:3000/dshin/scribblez.git"
    # Submodule: same host/owner, repo name from the submodule's origin basename,
    # needing no `gitea` remote of its own.
    assert publish.gitea_read_url(sup, "sub") == "http://localhost:3000/dshin/devenv_utils.git"


def make_synced_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A repo plus bare `gitea` and `origin` remotes, all at the same main."""
    repo = tmp_path / "repo"
    init_repo(repo)
    gitea = tmp_path / "gitea.git"
    git(tmp_path, "init", "-q", "--bare", str(gitea))
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "remote", "add", "gitea", str(gitea))
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "gitea", "main")
    git(repo, "push", "-q", "origin", "main")
    return repo, gitea, origin


def bare_main(anchor: Path, bare: Path) -> str:
    """The bare repo's main tip, read via ls-remote: agent harnesses set
    safe.bareRepository=explicit, which forbids running git *inside* a bare
    repo but not addressing it as a remote."""
    return git_out(anchor, "ls-remote", str(bare), "main").split()[0]


def advance_bare(tmp_path: Path, bare: Path, name: str, filename: str = ""):
    """Land a commit on the bare repo's main that the test repo doesn't have."""
    clone = tmp_path / f"clone-{name}"
    git(tmp_path, "clone", "-q", str(bare), str(clone))
    git(clone, "config", "user.name", "T")
    git(clone, "config", "user.email", "t@t")
    (clone / (filename or name)).write_text(name)
    git(clone, "add", filename or name)
    git(clone, "commit", "-q", "-m", name)
    git(clone, "push", "-q", "origin", "main")


def local_commit(repo: Path, name: str = "local", filename: str = ""):
    (repo / (filename or name)).write_text(name)
    git(repo, "add", filename or name)
    git(repo, "commit", "-q", "-m", name)


def feed_answers(monkeypatch, *answers: str) -> list:
    """Answer sync_main's prompts from a fixed script; returns the list the
    prompt texts are appended to, for asserting which prompts fired."""
    prompts = []
    answer_iter = iter(answers)

    def scripted_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answer_iter)

    monkeypatch.setattr("builtins.input", scripted_input)
    return prompts


def forbid_prompts(monkeypatch):
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.fail(f"unexpected prompt: {prompt}")
    )


def test_sync_main_fast_forwards_when_behind(tmp_path: Path, monkeypatch):
    repo, gitea, _ = make_synced_repo(tmp_path)
    forbid_prompts(monkeypatch)
    advance_bare(tmp_path, gitea, "merged")
    publish.sync_main(repo)
    assert git_out(repo, "rev-parse", "main") == bare_main(repo, gitea)


def test_sync_main_syncs_gitea_when_ahead(tmp_path: Path, monkeypatch):
    # A local-only main commit (its commit_guard mirror push never landed) is
    # a guaranteed fast-forward for Gitea: publish syncs it instead of
    # bouncing the user.
    repo, gitea, _ = make_synced_repo(tmp_path)
    forbid_prompts(monkeypatch)
    local_commit(repo)
    publish.sync_main(repo)
    assert bare_main(repo, gitea) == git_out(repo, "rev-parse", "main")


def test_sync_main_rebases_private_diverged_commits(tmp_path: Path, monkeypatch):
    repo, gitea, _ = make_synced_repo(tmp_path)
    advance_bare(tmp_path, gitea, "merged")
    local_commit(repo)
    prompts = feed_answers(monkeypatch, "")  # empty answer = default yes
    publish.sync_main(repo)
    assert len(prompts) == 1 and "Rebase" in prompts[0]
    # Linear history: the local commit now sits atop the Gitea commit.
    assert git_out(repo, "rev-list", "--merges", "main") == ""
    assert git_out(repo, "log", "--format=%s", "-2", "main").splitlines() == ["local", "merged"]
    assert bare_main(repo, gitea) == git_out(repo, "rev-parse", "main")


def test_sync_main_merges_when_local_only_commits_are_published(tmp_path: Path, monkeypatch):
    repo, gitea, origin = make_synced_repo(tmp_path)
    local_commit(repo)
    git(repo, "push", "-q", "origin", "main")
    published = git_out(repo, "rev-parse", "main")
    advance_bare(tmp_path, gitea, "merged")
    prompts = feed_answers(monkeypatch, "")
    publish.sync_main(repo)
    assert len(prompts) == 1 and "Merge Gitea" in prompts[0]
    # The published commit keeps its hash: merged, not rebased.
    assert publish.is_ancestor(repo, published, "main")
    assert git_out(repo, "rev-list", "--merges", "--count", "main") == "1"
    assert bare_main(repo, gitea) == git_out(repo, "rev-parse", "main")


def test_sync_main_merges_github_only_commits(tmp_path: Path, monkeypatch):
    repo, gitea, origin = make_synced_repo(tmp_path)
    advance_bare(tmp_path, origin, "outside")
    prompts = feed_answers(monkeypatch, "")
    publish.sync_main(repo)
    assert len(prompts) == 1 and "Merge them into your main" in prompts[0]
    assert publish.is_ancestor(repo, bare_main(repo, origin), "main")
    assert bare_main(repo, gitea) == git_out(repo, "rev-parse", "main")


def test_sync_main_reconciles_gitea_and_github_in_one_run(tmp_path: Path, monkeypatch):
    repo, gitea, origin = make_synced_repo(tmp_path)
    advance_bare(tmp_path, gitea, "merged")
    advance_bare(tmp_path, origin, "outside")
    local_commit(repo)
    prompts = feed_answers(monkeypatch, "", "")
    publish.sync_main(repo)
    assert len(prompts) == 2
    assert "Rebase" in prompts[0] and "Merge them into your main" in prompts[1]
    main_tip = git_out(repo, "rev-parse", "main")
    assert bare_main(repo, gitea) == main_tip
    assert publish.is_ancestor(repo, bare_main(repo, origin), "main")
    assert "local" in git_out(repo, "log", "--format=%s", "main").splitlines()


def test_sync_main_question_mark_explains_then_asks_again(tmp_path: Path, monkeypatch, capsys):
    repo, gitea, _ = make_synced_repo(tmp_path)
    advance_bare(tmp_path, gitea, "merged")
    local_commit(repo)
    prompts = feed_answers(monkeypatch, "?", "")
    publish.sync_main(repo)
    assert len(prompts) == 2
    assert "git rebase" in capsys.readouterr().out
    assert bare_main(repo, gitea) == git_out(repo, "rev-parse", "main")


def test_sync_main_decline_changes_nothing(tmp_path: Path, monkeypatch):
    repo, gitea, _ = make_synced_repo(tmp_path)
    advance_bare(tmp_path, gitea, "merged")
    local_commit(repo)
    before = git_out(repo, "rev-parse", "main")
    feed_answers(monkeypatch, "n")
    with pytest.raises(SystemExit, match="Nothing was changed"):
        publish.sync_main(repo)
    assert git_out(repo, "rev-parse", "main") == before


def test_sync_main_aborts_cleanly_on_rebase_conflict(tmp_path: Path, monkeypatch):
    repo, gitea, _ = make_synced_repo(tmp_path)
    # Both sides edit the same file: the rebase cannot apply cleanly.
    advance_bare(tmp_path, gitea, "gitea-side", filename="f")
    local_commit(repo, "local-side", filename="f")
    before = git_out(repo, "rev-parse", "main")
    feed_answers(monkeypatch, "")
    with pytest.raises(SystemExit, match="conflicts"):
        publish.sync_main(repo)
    assert git_out(repo, "rev-parse", "main") == before
    assert git_out(repo, "status", "--porcelain") == ""


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
