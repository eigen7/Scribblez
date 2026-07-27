"""Tests for the manual pointer-bump script (update_submodules.py).

The script sources each submodule's freshness tip from its GitHub `origin`, uses
Gitea to detect unpublished merges and to fast-forward a lagging Gitea main, and
only acts from a fully-published superproject on its remote head. The fixtures
build that world out of throwaway git repos: a superproject with bare `gitea`
and `origin` stand-ins at a shared main, and submodules whose own bare `origin`
and `gitea` stand-ins are seeded independently at chosen commits. No Gitea
service, no host.
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


def bare_main(anchor: Path, bare_repo: Path) -> str:
    return git_out(anchor, "ls-remote", str(bare_repo), "main").split()[0]


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


def make_sub(tmp_path: Path, name: str, origin_ref: str, gitea_ref: str) -> tuple[Path, Path, dict]:
    """Bare `origin` (GitHub) and `gitea` stand-ins for submodule `name`, seeded
    from a work repo with commits A (base), B (child of A), and C (a sibling
    child of A). origin's main is `origin_ref`; gitea's is `gitea_ref` (each one
    of 'A'/'B'/'C'). Returns (origin, gitea, {ref: sha})."""
    work = tmp_path / f"{name}_work"
    init_repo(work, "A")
    a_sha = git_out(work, "rev-parse", "HEAD")
    (work / "g").write_text("g")
    git(work, "add", "g")
    git(work, "commit", "-q", "-m", "B")
    b_sha = git_out(work, "rev-parse", "HEAD")
    git(work, "checkout", "-q", a_sha)
    (work / "h").write_text("h")
    git(work, "add", "h")
    git(work, "commit", "-q", "-m", "C")
    c_sha = git_out(work, "rev-parse", "HEAD")
    sha = {"A": a_sha, "B": b_sha, "C": c_sha}
    origin = bare(tmp_path, f"{name}_origin.git")
    gitea = bare(tmp_path, f"{name}_gitea.git")
    git(work, "push", "-q", str(origin), f"{sha[origin_ref]}:refs/heads/main")
    git(work, "push", "-q", str(gitea), f"{sha[gitea_ref]}:refs/heads/main")
    return origin, gitea, sha


def build_super(tmp_path, monkeypatch, sub_specs) -> tuple[Path, dict]:
    """A synced superproject with the given submodules. Each spec is
    (name, recorded_ref, origin_ref, gitea_ref). Returns (super, {name: (origin,
    gitea, sha_map)})."""
    super_ = tmp_path / "super"
    init_repo(super_, "super")
    info = {}
    gitea_map = {}
    for name, recorded_ref, origin_ref, gitea_ref in sub_specs:
        origin, gitea, sha = make_sub(tmp_path, name, origin_ref, gitea_ref)
        git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(origin), name)
        git(super_ / name, "checkout", "-q", sha[recorded_ref])
        git(super_, "add", name)
        git(super_, "commit", "-q", "-m", f"pin {name} at {recorded_ref}")
        # The sub's `gitea` remote is the heal-push target.
        git(super_ / name, "remote", "add", "gitea", str(gitea))
        info[name] = (origin, gitea, sha)
        gitea_map[name] = gitea
    super_gitea = bare(tmp_path, "super_gitea.git")
    super_origin = bare(tmp_path, "super_origin.git")
    git(super_, "remote", "add", "gitea", str(super_gitea))
    git(super_, "remote", "add", "origin", str(super_origin))
    git(super_, "push", "-q", "gitea", "main")
    git(super_, "push", "-q", "origin", "main")

    # Freshness reads of a submodule's Gitea URL resolve to its stand-in bare;
    # the superproject's own gitea remote (empty sub_path) is read as configured.
    def fake(root, sub_path=""):
        return str(gitea_map[sub_path]) if sub_path else str(super_gitea)

    monkeypatch.setattr(submodule_bump, "gitea_read_url", fake)
    return super_, info


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
    local_commit(super_)
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


def test_origin_ahead_heals_gitea_then_bumps(tmp_path: Path, monkeypatch, capsys):
    # David's case: origin (B) is ahead of the recorded pointer (A) and of the
    # lagging Gitea main (A). Gitea is fast-forwarded to origin, then the bump to
    # origin is offered and accepted.
    super_, info = build_super(tmp_path, monkeypatch, [("sub", "A", "B", "A")])
    _, gitea, sha = info["sub"]
    feed_answers(monkeypatch, "")  # default yes

    assert update_submodules.update_submodules(super_) == 0
    assert bare_main(super_, gitea) == sha["B"]  # Gitea caught up to origin
    assert sub_pointer(super_, "sub") == sha["B"]
    out = capsys.readouterr().out
    assert "fast-forwarded the submodule's Gitea main to origin" in out
    assert "git publish" in out


def test_gitea_ahead_unpublished_rejected(tmp_path: Path, monkeypatch, capsys):
    # Gitea (B) has a merge origin (A) lacks: unpublished, so `git publish`'s job.
    super_, info = build_super(tmp_path, monkeypatch, [("sub", "A", "A", "B")])
    _, gitea, sha = info["sub"]
    before = git_out(super_, "rev-parse", "HEAD")
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    err = capsys.readouterr().err
    assert "not pushed to origin yet" in err and "git publish" in err
    assert git_out(super_, "rev-parse", "HEAD") == before
    assert bare_main(super_, gitea) == sha["B"]  # Gitea untouched (no heal push)


def test_all_equal_up_to_date(tmp_path: Path, monkeypatch, capsys):
    super_, _ = build_super(tmp_path, monkeypatch, [("sub", "A", "A", "A")])
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 0
    out = capsys.readouterr().out
    assert "sub: up to date" in out
    assert "git publish" not in out


def test_gitea_origin_diverged_rejected(tmp_path: Path, monkeypatch, capsys):
    # Gitea (C) and origin (B) each hold commits the other lacks.
    super_, _ = build_super(tmp_path, monkeypatch, [("sub", "A", "B", "C")])
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    err = capsys.readouterr().err
    assert "diverged" in err and "manually" in err


def test_gitea_unreachable_rejected(tmp_path: Path, monkeypatch, capsys):
    super_, _ = build_super(tmp_path, monkeypatch, [("sub", "A", "B", "A")])
    # Point only the submodule's Gitea URL at a nonexistent path; the
    # superproject's gitea read (empty sub_path) still resolves.
    monkeypatch.setattr(
        submodule_bump,
        "gitea_read_url",
        lambda root, sub_path="": (
            str(tmp_path / "sub_gitea_gone") if sub_path else str(tmp_path / "super_gitea.git")
        ),
    )
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "Gitea repo could not be reached" in capsys.readouterr().err


def test_origin_unreachable_rejected(tmp_path: Path, monkeypatch, capsys):
    super_, _ = build_super(tmp_path, monkeypatch, [("sub", "A", "B", "A")])
    git(super_ / "sub", "remote", "set-url", "origin", str(tmp_path / "sub_origin_gone"))
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 1
    assert "origin could not be reached" in capsys.readouterr().err


def test_gitea_heal_push_failure_warns_but_bumps(tmp_path: Path, monkeypatch, capsys):
    # origin ahead of a lagging Gitea, but the heal push target is broken: the
    # bump is still valid (origin is the authority), so it proceeds with a warning.
    super_, info = build_super(tmp_path, monkeypatch, [("sub", "A", "B", "A")])
    _, _, sha = info["sub"]
    git(super_ / "sub", "remote", "set-url", "gitea", str(tmp_path / "heal_target_gone"))
    feed_answers(monkeypatch, "")

    assert update_submodules.update_submodules(super_) == 0
    assert "could not fast-forward the submodule's Gitea main" in capsys.readouterr().err
    assert sub_pointer(super_, "sub") == sha["B"]


def test_loop_continues_across_submodules(tmp_path: Path, monkeypatch, capsys):
    super_, _ = build_super(
        tmp_path, monkeypatch, [("asub", "A", "B", "B"), ("bsub", "A", "B", "B")]
    )
    feed_answers(monkeypatch, "n", "n")  # decline both

    assert update_submodules.update_submodules(super_) == 0
    out = capsys.readouterr().out
    assert "asub: submodule upstream is ahead" in out
    assert "bsub: submodule upstream is ahead" in out


# ---- content-free (identical-tree) tip suppression ----------------------


def test_bump_status_suppresses_content_free_merge(tmp_path: Path):
    # A --no-ff merge of an already-contained branch: strictly ahead of the
    # pre-merge tip, but with an identical tree -- nothing to bump to.
    sub = tmp_path / "sub"
    init_repo(sub, "base")
    git(sub, "checkout", "-q", "-b", "feature")
    (sub / "g").write_text("g")
    git(sub, "add", "g")
    git(sub, "commit", "-q", "-m", "real work")
    pre_merge = git_out(sub, "rev-parse", "HEAD")
    git(sub, "checkout", "-q", "main")
    git(sub, "merge", "--no-ff", "-m", "Merge feature", "feature")
    merge = git_out(sub, "rev-parse", "HEAD")

    assert submodule_bump.trees_equal(sub, pre_merge, merge)
    assert submodule_bump.bump_status(sub, pre_merge, merge) == "none"


def test_bump_status_ready_for_real_content(tmp_path: Path):
    sub = tmp_path / "sub"
    init_repo(sub, "A")
    a_sha = git_out(sub, "rev-parse", "HEAD")
    (sub / "g").write_text("g")
    git(sub, "add", "g")
    git(sub, "commit", "-q", "-m", "B")
    b_sha = git_out(sub, "rev-parse", "HEAD")

    assert submodule_bump.bump_status(sub, a_sha, b_sha) == "ahead"


def build_super_content_free_merge(tmp_path: Path, monkeypatch) -> Path:
    """A synced super whose submodule's origin (and gitea) main is a content-free
    --no-ff merge over the recorded pointer (the pre-merge tip)."""
    work = tmp_path / "sub_work"
    init_repo(work, "base")
    git(work, "checkout", "-q", "-b", "feature")
    (work / "g").write_text("g")
    git(work, "add", "g")
    git(work, "commit", "-q", "-m", "real work")
    pre_merge = git_out(work, "rev-parse", "HEAD")
    git(work, "checkout", "-q", "main")
    git(work, "merge", "--no-ff", "-m", "Merge feature", "feature")
    merge = git_out(work, "rev-parse", "HEAD")
    sub_origin = bare(tmp_path, "sub_origin.git")
    sub_gitea = bare(tmp_path, "sub_gitea.git")
    git(work, "push", "-q", str(sub_origin), f"{merge}:refs/heads/main")
    git(work, "push", "-q", str(sub_gitea), f"{merge}:refs/heads/main")

    super_ = tmp_path / "super"
    init_repo(super_, "super")
    git(super_, "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), "sub")
    git(super_ / "sub", "checkout", "-q", pre_merge)
    git(super_, "add", "sub")
    git(super_, "commit", "-q", "-m", "pin sub at pre-merge tip")
    git(super_ / "sub", "remote", "add", "gitea", str(sub_gitea))
    super_gitea = bare(tmp_path, "super_gitea.git")
    super_origin = bare(tmp_path, "super_origin.git")
    git(super_, "remote", "add", "gitea", str(super_gitea))
    git(super_, "remote", "add", "origin", str(super_origin))
    git(super_, "push", "-q", "gitea", "main")
    git(super_, "push", "-q", "origin", "main")
    monkeypatch.setattr(
        submodule_bump,
        "gitea_read_url",
        lambda root, sub_path="": str(sub_gitea) if sub_path else str(super_gitea),
    )
    return super_


def test_content_free_merge_reports_up_to_date(tmp_path: Path, monkeypatch, capsys):
    super_ = build_super_content_free_merge(tmp_path, monkeypatch)
    forbid_prompts(monkeypatch)

    assert update_submodules.update_submodules(super_) == 0
    out = capsys.readouterr().out
    assert "sub: up to date" in out
    assert "git publish" not in out
