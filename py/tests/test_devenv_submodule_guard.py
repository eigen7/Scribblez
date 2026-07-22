"""Tests for the submodule pointer/sync hooks
(submodules/devenv_utils/submodule_guard.py).

The hook actions are exercised as plain functions against throwaway git
repos: a superproject holding a real submodule, with the submodule's source
repo standing in for its upstream. Hook installation itself is the wizard's
job and isn't covered here.
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import submodule_bump, submodule_guard  # noqa: E402


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


# ---- offer-update: the pull-time freshness reaction ----------------------
#
# `repos` pins `sub` at A while its upstream also has a later B. Pointing the
# submodule's derived Gitea URL at that upstream makes the freshness fetch read
# B as the "Gitea main" tip.


def point_gitea_at_upstream(monkeypatch, tmp_path: Path):
    upstream = tmp_path / "upstream"
    monkeypatch.setattr(submodule_bump, "gitea_read_url", lambda root, sub_path="": str(upstream))


def sub_pointer(super_: Path) -> str:
    return git_out(super_, "ls-tree", "HEAD", "sub").split()[2]


def write_devenv_toml(super_: Path, body: str = 'name = "proj"\n'):
    (super_ / "devenv.toml").write_text(body)


def test_react_never_prints_note_without_committing(repos, tmp_path, monkeypatch, capsys):
    super_, a_sha, b_sha = repos
    point_gitea_at_upstream(monkeypatch, tmp_path)
    before = git_out(super_, "rev-parse", "HEAD")

    submodule_guard.react_to_bump(super_, "sub", "sub", "never")

    assert git_out(super_, "rev-parse", "HEAD") == before
    assert sub_pointer(super_) == a_sha
    out = capsys.readouterr().out
    assert "new upstream commits" in out and b_sha[:7] in out


def test_react_always_bumps_without_prompting(repos, tmp_path, monkeypatch, capsys):
    super_, _, b_sha = repos
    point_gitea_at_upstream(monkeypatch, tmp_path)
    monkeypatch.setattr(
        submodule_guard, "open_tty", lambda: pytest.fail("always mode must not prompt")
    )

    submodule_guard.react_to_bump(super_, "sub", "sub", "always")

    assert sub_pointer(super_) == b_sha
    assert git_out(super_, "log", "--format=%s", "-1") == f"Bump sub submodule to {b_sha[:7]}"
    assert "Updated sub submodule" in capsys.readouterr().out


def test_react_prompt_no_tty_falls_back_to_note(repos, tmp_path, monkeypatch, capsys):
    super_, a_sha, _ = repos
    point_gitea_at_upstream(monkeypatch, tmp_path)
    monkeypatch.setattr(submodule_guard, "open_tty", lambda: None)

    submodule_guard.react_to_bump(super_, "sub", "sub", "prompt")

    assert sub_pointer(super_) == a_sha
    assert "new upstream commits" in capsys.readouterr().out


def test_react_prompt_accept_commits_bump(repos, tmp_path, monkeypatch):
    super_, _, b_sha = repos
    point_gitea_at_upstream(monkeypatch, tmp_path)
    monkeypatch.setattr(submodule_guard, "open_tty", lambda: io.StringIO())
    monkeypatch.setattr(submodule_guard, "tty_prompt", lambda tty, q, e: True)

    submodule_guard.react_to_bump(super_, "sub", "sub", "prompt")

    assert sub_pointer(super_) == b_sha
    assert git_out(super_, "log", "--format=%s", "-1") == f"Bump sub submodule to {b_sha[:7]}"


def test_react_prompt_decline_then_save_writes_local_toml(repos, tmp_path, monkeypatch, capsys):
    super_, a_sha, _ = repos
    point_gitea_at_upstream(monkeypatch, tmp_path)
    monkeypatch.setattr(submodule_guard, "open_tty", lambda: io.StringIO())
    # First prompt (bump) declined, second prompt (save selection) accepted.
    answers = iter([False, True])
    monkeypatch.setattr(submodule_guard, "tty_prompt", lambda tty, q, e: next(answers))
    before = git_out(super_, "rev-parse", "HEAD")

    submodule_guard.react_to_bump(super_, "sub", "sub", "prompt")

    assert git_out(super_, "rev-parse", "HEAD") == before  # no bump commit
    assert sub_pointer(super_) == a_sha
    # The selection lands in the untracked devenv.local.toml, created for it.
    local = super_ / "devenv.local.toml"
    assert local.exists()
    assert "[submodules]" in local.read_text()
    assert 'pull_update = "never"' in local.read_text()
    assert not (super_ / "devenv.toml").exists()  # tracked config untouched
    assert "devenv.local.toml" in capsys.readouterr().out


def test_react_gitea_unreachable_is_silent(repos, tmp_path, monkeypatch, capsys):
    super_, a_sha, _ = repos
    # A URL whose fetch fails: submodule_gitea_tip returns None, so nothing fires.
    monkeypatch.setattr(
        submodule_bump, "gitea_read_url", lambda root, sub_path="": str(tmp_path / "gone.git")
    )
    before = git_out(super_, "rev-parse", "HEAD")

    submodule_guard.react_to_bump(super_, "sub", "sub", "prompt")

    assert git_out(super_, "rev-parse", "HEAD") == before
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def add_gitea_remote(super_: Path, tmp_path: Path) -> Path:
    bare = tmp_path / "super-gitea.git"
    git(tmp_path, "init", "-q", "--bare", str(bare))
    git(super_, "remote", "add", "gitea", str(bare))
    git(super_, "push", "-q", "gitea", "main")
    return bare


def test_offer_update_noop_off_main(repos, tmp_path, monkeypatch, capsys):
    super_, a_sha, _ = repos
    add_gitea_remote(super_, tmp_path)
    point_gitea_at_upstream(monkeypatch, tmp_path)
    git(super_, "checkout", "-q", "-b", "feature")

    submodule_guard.offer_update(super_)

    assert sub_pointer(super_) == a_sha
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_offer_update_on_main_reacts_per_mode(repos, tmp_path, monkeypatch, capsys):
    super_, _, b_sha = repos
    add_gitea_remote(super_, tmp_path)
    point_gitea_at_upstream(monkeypatch, tmp_path)
    write_devenv_toml(super_, 'name = "proj"\n[submodules]\npull_update = "always"\n')

    submodule_guard.offer_update(super_)

    assert sub_pointer(super_) == b_sha
    assert "Updated sub submodule" in capsys.readouterr().out
