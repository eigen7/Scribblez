"""Tests for the manual pointer-bump script (update_submodules.py).

The script only acts from a fully-published superproject sitting at its remote
head, and only bumps to submodule commits already on the submodule's GitHub
origin. The fixtures build that world out of throwaway git repos: a superproject
with bare `gitea` and `origin` stand-ins at a shared main, and submodules whose
own bare `origin` (GitHub) and `gitea` stand-ins are seeded independently so a
tip can be "on gitea but not origin". No Gitea service, no host.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils import submodule_bump, update_submodules  # noqa: E402


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


def bare(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(p))
    return p


def feed_answers(monkeypatch, *answers: str):
    answer_iter = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answer_iter))


def forbid_prompts(monkeypatch):
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.fail(f"unexpected prompt: {prompt}")
    )


def local_commit(repo: Path, name: str = "local"):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)


def advance_bare(tmp_path: Path, bare_repo: Path, name: str):
    """Land a commit on the bare repo's main that the test repo lacks."""
    clone = tmp_path / f"clone_{name}"
    git(tmp_path, "clone", "-q", str(bare_repo), str(clone))
    git(clone, "config", "user.name", "T")
    git(clone, "config", "user.email", "t@t")
    (clone / name).write_text(name)
    git(clone, "add", name)
    git(clone, "commit", "-q", "-m", name)
    git(clone, "push", "-q", "origin", "main")


def sub_pointer(super_: Path, name: str) -> str:
    return git_out(super_, "ls-tree", "HEAD", name).split()[2]


# ---- superproject fixtures ----------------------------------------------


def make_synced_super(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A superproject on main with bare `gitea` and `origin` remotes all at the
    same main -- the fully-published state the script requires."""
    super_ = tmp_path / "super"
    init_repo(super_, "super")
    gitea = bare(tmp_path, "super_gitea.git")
    origin = bare(tmp_path, "super_origin.git")
    git(super_, "remote", "add", "gitea", str(gitea))
    git(super_, "remote", "add", "origin", str(origin))
    git(super_, "push", "-q", "gitea", "main")
    git(super_, "push", "-q", "origin", "main")
    return super_, gitea, origin


def make_sub_bares(tmp_path: Path, name: str, *, origin_has_b: bool) -> tuple[Path, Path, str, str]:
    """Bare `origin` (GitHub) and `gitea` stand-ins for submodule `name`, with
    commits A and B. gitea's main is B; origin's main is B when published, else
    A. Returns (origin, gitea, A, B)."""
    work = tmp_path / f"{name}_work"
    init_repo(work, "A")
    (work / "g").write_text("g")
    git(work, "add", "g")
    git(work, "commit", "-q", "-m", "B")
    a_sha = git_out(work, "rev-parse", "HEAD~1")
    b_sha = git_out(work, "rev-parse", "HEAD")
    origin = bare(tmp_path, f"{name}_origin.git")
    gitea = bare(tmp_path, f"{name}_gitea.git")
    git(work, "push", "-q", str(origin), f"{b_sha if origin_has_b else a_sha}:refs/heads/main")
    git(work, "push", "-q", str(gitea), f"{b_sha}:refs/heads/main")
    return origin, gitea, a_sha, b_sha


def build_super(tmp_path, monkeypatch, sub_specs) -> tuple[Path, dict, Path, Path]:
    """A synced superproject with the given submodules. Each spec is
    (name, published, origin_ok, gitea_ok). Returns (super, {name: (A, B)},
    super_gitea, super_origin)."""
    super_ = tmp_path / "super"
    init_repo(super_, "super")
    gitea_map = {}
    shas = {}
    for name, published, origin_ok, gitea_ok in sub_specs:
        sub_origin, sub_gitea, a_sha, b_sha = make_sub_bares(tmp_path, name, origin_has_b=published)
        git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), name)
        git(super_ / name, "checkout", "-q", a_sha)
        git(super_, "add", name)
        git(super_, "commit", "-q", "-m", f"pin {name} at A")
        if not origin_ok:
            git(super_ / name, "remote", "set-url", "origin", str(tmp_path / f"{name}_origin_gone"))
        gitea_map[name] = sub_gitea if gitea_ok else tmp_path / f"{name}_gitea_gone"
        shas[name] = (a_sha, b_sha)
    super_gitea = bare(tmp_path, "super_gitea.git")
    super_origin = bare(tmp_path, "super_origin.git")
    git(super_, "remote", "add", "gitea", str(super_gitea))
    git(super_, "remote", "add", "origin", str(super_origin))
    git(super_, "push", "-q", "gitea", "main")
    git(super_, "push", "-q", "origin", "main")

    # The superproject preconditions read the real gitea remote (above); only the
    # per-submodule Gitea URL derivation is redirected to the stand-in bares.
    def fake(root, sub_path=""):
        return str(gitea_map[sub_path]) if sub_path else str(super_gitea)

    monkeypatch.setattr(submodule_bump, "gitea_read_url", fake)
    return super_, shas, super_gitea, super_origin


# ---- whole-run precondition rejections ----------------------------------


def test_reject_wrong_branch(tmp_path: Path, monkeypatch, capsys):
    super_, _, _ = make_synced_super(tmp_path)
    git(super_, "checkout", "-q", "-b", "feature")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    err = capsys.readouterr().err
    assert "not on branch main" in err and "git publish" in err


def test_reject_dirty_tree(tmp_path: Path, monkeypatch, capsys):
    super_, _, _ = make_synced_super(tmp_path)
    (super_ / "f").write_text("edited")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "uncommitted changes" in capsys.readouterr().err


def test_reject_local_ahead_of_gitea(tmp_path: Path, monkeypatch, capsys):
    super_, _, _ = make_synced_super(tmp_path)
    local_commit(super_)  # not pushed anywhere
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "not yet on Gitea" in capsys.readouterr().err


def test_reject_local_behind_gitea(tmp_path: Path, monkeypatch, capsys):
    super_, gitea, _ = make_synced_super(tmp_path)
    advance_bare(tmp_path, gitea, "gitea-merge")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "Gitea has merges the local main lacks" in capsys.readouterr().err


def test_reject_local_not_on_origin(tmp_path: Path, monkeypatch, capsys):
    super_, _, _ = make_synced_super(tmp_path)
    local_commit(super_)
    git(super_, "push", "-q", "gitea", "main")  # gitea caught up, origin not
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "not yet on GitHub origin" in capsys.readouterr().err


def test_reject_origin_unreachable(tmp_path: Path, monkeypatch, capsys):
    super_, _, _ = make_synced_super(tmp_path)
    git(super_, "remote", "set-url", "origin", str(tmp_path / "origin_gone"))
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "GitHub origin could not be reached" in capsys.readouterr().err


# ---- per-submodule handling ---------------------------------------------


def test_published_tip_accept_commits_bump(tmp_path: Path, monkeypatch, capsys):
    super_, shas, _, _ = build_super(tmp_path, monkeypatch, [("sub", True, True, True)])
    _, b_sha = shas["sub"]
    feed_answers(monkeypatch, "")  # default yes

    assert update_submodules.update_submodules(super_) == 0
    assert sub_pointer(super_, "sub") == b_sha
    assert git_out(super_, "log", "--format=%s", "-1") == f"Bump sub submodule to {b_sha[:7]}"
    assert "git publish" in capsys.readouterr().out


def test_published_tip_decline_no_commit(tmp_path: Path, monkeypatch, capsys):
    super_, shas, _, _ = build_super(tmp_path, monkeypatch, [("sub", True, True, True)])
    a_sha, _ = shas["sub"]
    before = git_out(super_, "rev-parse", "HEAD")
    feed_answers(monkeypatch, "n")

    assert update_submodules.update_submodules(super_) == 0
    assert git_out(super_, "rev-parse", "HEAD") == before
    assert sub_pointer(super_, "sub") == a_sha
    assert "git publish" not in capsys.readouterr().out


def test_up_to_date_prints_line(tmp_path: Path, monkeypatch, capsys):
    super_, shas, super_gitea, _ = build_super(tmp_path, monkeypatch, [("sub", True, True, True)])
    _, b_sha = shas["sub"]
    # Advance the pointer to the tip, then republish main so the preconditions
    # still hold: now nothing is ahead.
    git(super_ / "sub", "checkout", "-q", b_sha)
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin sub at B")
    git(super_, "push", "-q", str(super_gitea), "main")
    git(super_, "push", "-q", "origin", "main")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 0
    out = capsys.readouterr().out
    assert "sub: up to date" in out
    assert "git publish" not in out


def test_diverged_warns(tmp_path: Path, monkeypatch, capsys):
    super_, shas, super_gitea, _ = build_super(tmp_path, monkeypatch, [("sub", True, True, True)])
    a_sha, _ = shas["sub"]
    # Pin at a private commit C forking from A, diverging from the gitea tip B.
    git(super_ / "sub", "checkout", "-q", a_sha)
    (super_ / "sub" / "c").write_text("c")
    git(super_ / "sub", "add", "c")
    git(super_ / "sub", "commit", "-q", "-m", "C (private)")
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin sub at C")
    git(super_, "push", "-q", str(super_gitea), "main")
    git(super_, "push", "-q", "origin", "main")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 0
    assert "diverged" in capsys.readouterr().err


def test_unpublished_tip_refused(tmp_path: Path, monkeypatch, capsys):
    # gitea has B, origin does not: the merge is unpublished, so `git publish`'s
    # job, not this script's.
    super_, _, _, _ = build_super(tmp_path, monkeypatch, [("sub", False, True, True)])
    before = git_out(super_, "rev-parse", "HEAD")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    err = capsys.readouterr().err
    assert "not pushed to origin yet" in err and "git publish" in err
    assert git_out(super_, "rev-parse", "HEAD") == before


def test_sub_origin_unreachable_refused(tmp_path: Path, monkeypatch, capsys):
    super_, _, _, _ = build_super(tmp_path, monkeypatch, [("sub", True, False, True)])
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "origin could not be reached" in capsys.readouterr().err


def test_sub_gitea_unreachable_refused(tmp_path: Path, monkeypatch, capsys):
    super_, _, _, _ = build_super(tmp_path, monkeypatch, [("sub", True, True, False)])
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "Gitea repo could not be reached" in capsys.readouterr().err


def test_loop_continues_across_submodules(tmp_path: Path, monkeypatch, capsys):
    super_, _, _, _ = build_super(
        tmp_path, monkeypatch, [("asub", True, True, True), ("bsub", True, True, True)]
    )
    feed_answers(monkeypatch, "n", "n")  # decline both

    assert update_submodules.update_submodules(super_) == 0
    out = capsys.readouterr().out
    # Both submodules were reached and offered.
    assert "asub: submodule Gitea main is ahead" in out
    assert "bsub: submodule Gitea main is ahead" in out
