"""Tests for the Gitea client layer (submodules/devenv_utils/gitea_client.py).

Pure URL construction, the env contract, and remote management against a
throwaway git repo -- no Gitea server, no Docker.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# submodules.* lives at the repo root, not on the py/-rooted PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submodules.devenv_utils.gitea_client import (  # noqa: E402
    BACKEND_URL_ENV,
    WEB_PORT_ENV,
    WEB_URL_ENV,
    GiteaAccess,
    container_access,
    ensure_remote,
)


def git(cwd: Path, *args: str):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def access() -> GiteaAccess:
    """A container-side-shaped access: service names for reachability, the
    host web port for canonical/browser URLs."""
    return GiteaAccess(
        web_url="http://devenv-gitea:3000",
        backend_url="http://devenv-gitea:3001",
        host_web_port=3100,
        credentials_dir=Path("/nonexistent"),
    )


def test_canonical_url_is_host_shaped_and_credential_free(access: GiteaAccess):
    # The canonical remote URL must resolve on the host as-is (the container
    # side reaches it via the system-gitconfig rewrite), so it carries the
    # host web port -- not the in-container service URL.
    assert access.canonical_repo_url("dshin", "scribblez") == (
        "http://localhost:3100/dshin/scribblez.git"
    )


def test_browser_url_targets_host_port(access: GiteaAccess):
    assert access.browser_url("dshin/scribblez/pulls/7") == (
        "http://localhost:3100/dshin/scribblez/pulls/7"
    )


def test_authed_repo_url_embeds_credentials_in_backend_url(access: GiteaAccess):
    url = access.authed_repo_url("dshin", "scribblez", {"username": "claude", "password": "pw"})
    assert url == "http://claude:pw@devenv-gitea:3001/dshin/scribblez.git"


def test_read_repo_url_is_anonymous_backend(access: GiteaAccess):
    assert access.read_repo_url("dshin", "x") == "http://devenv-gitea:3001/dshin/x.git"


def test_container_access_reads_env_contract(monkeypatch):
    monkeypatch.setenv(WEB_PORT_ENV, "3000")
    monkeypatch.setenv(WEB_URL_ENV, "http://devenv-gitea:3000")
    monkeypatch.setenv(BACKEND_URL_ENV, "http://devenv-gitea:3001")
    access = container_access()
    assert access.host_web_port == 3000
    assert access.backend_url == "http://devenv-gitea:3001"


def test_container_access_requires_env_contract(monkeypatch):
    # A container launched before the Gitea service setup lacks the env vars;
    # the failure must carry re-setup guidance rather than a KeyError.
    monkeypatch.delenv(WEB_PORT_ENV, raising=False)
    monkeypatch.delenv(WEB_URL_ENV, raising=False)
    monkeypatch.delenv(BACKEND_URL_ENV, raising=False)
    with pytest.raises(SystemExit, match="setup_wizard"):
        container_access()


def test_ensure_remote_adds_then_updates(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    ensure_remote(repo, "http://localhost:3000/dshin/x.git")
    assert git_out(repo, "remote", "get-url", "gitea") == "http://localhost:3000/dshin/x.git"
    # An existing remote with a stale URL (e.g. the credential-embedded shape
    # of a pre-service checkout) is updated in place.
    ensure_remote(repo, "http://localhost:3100/dshin/x.git")
    assert git_out(repo, "remote", "get-url", "gitea") == "http://localhost:3100/dshin/x.git"
