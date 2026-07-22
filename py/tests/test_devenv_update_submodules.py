"""Tests for the manual pointer-bump script (update_submodules.py).

The script is exercised as plain functions against throwaway git repos: a
superproject holding a real submodule, with a bare repo standing in for the
submodule's Gitea main. No Gitea, no host. The interactive prompt is the same
publish.confirm path the publish tests drive, so builtins.input is scripted the
same way.
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


def add_submodule(super_: Path, tmp_path: Path, name: str) -> tuple[Path, str, str]:
    """Add submodule `name` to `super_`, pinned at commit A, whose upstream (the
    Gitea stand-in) also has a later commit B. Returns (upstream, A, B)."""
    upstream = tmp_path / f"up_{name}"
    init_repo(upstream, "A")
    (upstream / "g").write_text("g")
    git(upstream, "add", "g")
    git(upstream, "commit", "-q", "-m", "add g (B)")
    a_sha = git_out(upstream, "rev-parse", "HEAD~1")
    b_sha = git_out(upstream, "rev-parse", "HEAD")

    git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(upstream), name)
    git(super_ / name, "checkout", "-q", a_sha)
    git(super_, "add", name)
    git(super_, "commit", "-q", "-m", f"pin {name} at A")
    return upstream, a_sha, b_sha


def make_super(tmp_path: Path) -> Path:
    super_ = tmp_path / "super"
    init_repo(super_, "super")
    return super_


def point_gitea(monkeypatch, mapping: dict):
    """Make each submodule's derived Gitea URL resolve to a stand-in upstream,
    keyed by sub_path."""
    monkeypatch.setattr(
        submodule_bump, "gitea_read_url", lambda root, sub_path="": str(mapping[sub_path])
    )


def feed_answers(monkeypatch, *answers: str):
    answer_iter = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answer_iter))


def forbid_prompts(monkeypatch):
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.fail(f"unexpected prompt: {prompt}")
    )


def sub_pointer(super_: Path, name: str) -> str:
    return git_out(super_, "ls-tree", "HEAD", name).split()[2]


def test_accept_commits_bump_and_closes(tmp_path: Path, monkeypatch, capsys):
    super_ = make_super(tmp_path)
    upstream, _, b_sha = add_submodule(super_, tmp_path, "sub")
    point_gitea(monkeypatch, {"sub": upstream})
    feed_answers(monkeypatch, "")  # default yes

    update_submodules.update_submodules(super_)

    assert sub_pointer(super_, "sub") == b_sha
    assert git_out(super_, "log", "--format=%s", "-1") == f"Bump sub submodule to {b_sha[:7]}"
    assert "git publish" in capsys.readouterr().out


def test_decline_leaves_pointer_and_no_closing(tmp_path: Path, monkeypatch, capsys):
    super_ = make_super(tmp_path)
    upstream, a_sha, _ = add_submodule(super_, tmp_path, "sub")
    point_gitea(monkeypatch, {"sub": upstream})
    before = git_out(super_, "rev-parse", "HEAD")
    feed_answers(monkeypatch, "n")

    update_submodules.update_submodules(super_)

    assert git_out(super_, "rev-parse", "HEAD") == before
    assert sub_pointer(super_, "sub") == a_sha
    assert "git publish" not in capsys.readouterr().out


def test_up_to_date_prints_line_no_closing(tmp_path: Path, monkeypatch, capsys):
    super_ = make_super(tmp_path)
    upstream, _, b_sha = add_submodule(super_, tmp_path, "sub")
    # Pin the pointer at B, matching the upstream tip: nothing to offer.
    git(super_ / "sub", "checkout", "-q", b_sha)
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin sub at B")
    point_gitea(monkeypatch, {"sub": upstream})
    forbid_prompts(monkeypatch)

    update_submodules.update_submodules(super_)

    out = capsys.readouterr().out
    assert "sub: up to date" in out
    assert "git publish" not in out


def test_diverged_warns_no_closing(tmp_path: Path, monkeypatch, capsys):
    super_ = make_super(tmp_path)
    upstream, a_sha, _ = add_submodule(super_, tmp_path, "sub")
    # Pin at a private commit C that forks from A, diverging from upstream B.
    git(super_ / "sub", "checkout", "-q", a_sha)
    (super_ / "sub" / "c").write_text("c")
    git(super_ / "sub", "add", "c")
    git(super_ / "sub", "commit", "-q", "-m", "C (private)")
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin sub at C")
    point_gitea(monkeypatch, {"sub": upstream})
    forbid_prompts(monkeypatch)

    update_submodules.update_submodules(super_)

    captured = capsys.readouterr()
    assert "diverged" in captured.err
    assert "git publish" not in captured.out


def test_decline_continues_to_next_submodule(tmp_path: Path, monkeypatch, capsys):
    super_ = make_super(tmp_path)
    up_a, a_pin, _ = add_submodule(super_, tmp_path, "asub")  # ahead, will be declined
    up_b, _, b_sha = add_submodule(super_, tmp_path, "bsub")
    # Pin bsub at its tip so it reports up-to-date, proving the loop reached it.
    git(super_ / "bsub", "checkout", "-q", b_sha)
    git(super_, "add", "bsub")
    git(super_, "commit", "-q", "-m", "pin bsub at B")
    point_gitea(monkeypatch, {"asub": up_a, "bsub": up_b})
    feed_answers(monkeypatch, "n")  # decline the one prompt (asub)

    update_submodules.update_submodules(super_)

    assert sub_pointer(super_, "asub") == a_pin  # declined, unchanged
    out = capsys.readouterr().out
    assert "bsub: up to date" in out  # loop continued past the decline
    assert "git publish" not in out
