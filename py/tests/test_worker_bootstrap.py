"""The worker image's bootstrap (docker-setup/worker/bootstrap.py): it unpacks
the payload the dashboard copied into the container and hands off to the
bundle's worker entrypoint."""

import importlib.util
import io
import tarfile

import pytest
from scribblez.paths import REPO_ROOT


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    """The script as a module, its payload and repo directories under tmp_path,
    the arch probe and the hand-off stubbed."""
    spec = importlib.util.spec_from_file_location(
        "bootstrap", REPO_ROOT / "docker-setup" / "worker" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PAYLOAD_DIR", tmp_path / "payload")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(module, "detect_host_arch", lambda: "znver3")
    module.handed_off = []
    monkeypatch.setattr(module.os, "execv", lambda exe, argv: module.handed_off.append(argv))
    return module


def _tarball(path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_the_payload_is_unpacked_and_handed_off(bootstrap, monkeypatch):
    payload = bootstrap.PAYLOAD_DIR
    _tarball(payload / "bundle.tar.gz", {"py/cloud/worker_entrypoint.py": b"# entry"})
    _tarball(payload / "positions.tar.gz", {"positions/NWL23/small/pos-01.gcg": b"#"})
    monkeypatch.delenv("SCZ_HOST_ARCH", raising=False)
    bootstrap.main()
    repo = bootstrap.REPO_ROOT
    assert (repo / "py" / "cloud" / "worker_entrypoint.py").read_bytes() == b"# entry"
    assert (repo / "positions" / "NWL23" / "small" / "pos-01.gcg").exists()
    assert bootstrap.os.environ["SCZ_HOST_ARCH"] == "znver3"
    assert len(bootstrap.handed_off) == 1


def test_a_container_given_no_bundle_says_so(bootstrap):
    with pytest.raises(SystemExit, match="no bundle"):
        bootstrap.main()
    assert bootstrap.handed_off == []
