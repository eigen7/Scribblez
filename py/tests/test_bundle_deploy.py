"""Tests for bundle deployment: the source fingerprint, the deploy path that
keeps the store's LATEST equal to the controller's tree, and the store."""

import json
import os

from cloud import bundles


def _tree(tmp_path, **contents) -> list[tuple[str, object]]:
    files = []
    for name, text in contents.items():
        path = tmp_path / name
        path.write_text(text)
        files.append((name, path))
    return sorted(files)


def test_source_hash_is_stable_and_content_addressed(tmp_path, monkeypatch):
    files = _tree(tmp_path, a="one", b="two")
    monkeypatch.setattr(bundles, "_shipped_files", lambda archs: files)
    first = bundles.source_hash(["x86-64"])
    assert first == bundles.source_hash(["x86-64"])

    (tmp_path / "b").write_text("three")
    assert bundles.source_hash(["x86-64"]) != first


def test_source_hash_covers_the_name_a_file_ships_under(tmp_path, monkeypatch):
    """Two archs' binaries can hold identical bytes; swapping which is which
    is still a different bundle."""
    swapped = [(n, p) for n, p in _tree(tmp_path, a="x", b="y")]
    monkeypatch.setattr(bundles, "_shipped_files", lambda archs: swapped)
    straight = bundles.source_hash(["x86-64"])
    monkeypatch.setattr(
        bundles, "_shipped_files", lambda archs: [(n, p) for n, p in reversed(swapped)]
    )
    assert bundles.source_hash(["x86-64"]) != straight


def test_source_hash_is_unknown_while_an_arch_is_unbuilt(tmp_path, monkeypatch):
    files = _tree(tmp_path, a="one") + [("missing", tmp_path / "nope")]
    monkeypatch.setattr(bundles, "_shipped_files", lambda archs: files)
    assert bundles.source_hash(["x86-64"]) is None


def test_source_hash_cache_follows_the_file(tmp_path, monkeypatch):
    """The cache exists so a status poll costs a stat walk; it must not hand
    back a digest for content that has since changed."""
    files = _tree(tmp_path, a="one")
    monkeypatch.setattr(bundles, "_shipped_files", lambda archs: files)
    cache = {}
    before = bundles.source_hash(["x86-64"], cache)
    (tmp_path / "a").write_text("changed")
    assert bundles.source_hash(["x86-64"], cache) != before


def _manifest(source_hash: str, bundle_id: str = "old") -> bundles.BundleManifest:
    return bundles.BundleManifest(
        bundle_id=bundle_id, git_sha="s", git_dirty=False, archs=["x86-64"], source_hash=source_hash
    )


def _deploy_harness(monkeypatch, *, latest, local_hash="local"):
    calls = {"built": 0, "written": 0, "archs": []}

    def build(archs, jobs=None):
        calls["built"] += 1
        calls["archs"].append(archs)

    def write(store, archs):
        calls["written"] += 1
        return bundles.BundleManifest(
            bundle_id="new", git_sha="s", git_dirty=False, archs=archs, source_hash=local_hash
        )

    monkeypatch.setattr(bundles, "build_archs", build)
    monkeypatch.setattr(bundles, "source_hash", lambda archs, cache=None: local_hash)
    monkeypatch.setattr(bundles, "latest_manifest", lambda store: latest)
    monkeypatch.setattr(bundles, "write_bundle", write)
    return calls


def test_deploy_writes_when_the_store_is_behind(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=_manifest("stale"))
    assert bundles.deploy_current_tree(None, ["x86-64"]).bundle_id == "new"
    assert (calls["built"], calls["written"]) == (1, 1)


def test_deploy_skips_writing_when_the_store_already_has_this_tree(monkeypatch):
    """An id that changed on every deploy would unpin every task that shares
    it, replacing containers to run identical code."""
    calls = _deploy_harness(monkeypatch, latest=_manifest("local"))
    assert bundles.deploy_current_tree(None, ["x86-64"]).bundle_id == "old"
    assert (calls["built"], calls["written"]) == (1, 0)


def test_deploy_writes_when_nothing_has_ever_been_written(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=None)
    assert bundles.deploy_current_tree(None, ["x86-64"]).bundle_id == "new"
    assert calls["written"] == 1


def test_deploy_writes_over_a_manifest_that_predates_source_hashes(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=_manifest(""))
    assert bundles.deploy_current_tree(None, ["x86-64"]).bundle_id == "new"
    assert calls["written"] == 1


def test_deploy_always_builds_before_deciding(monkeypatch):
    """The fingerprint covers compiled binaries: deciding without building
    would ship an unbuilt arch under a current-looking id."""
    calls = _deploy_harness(monkeypatch, latest=_manifest("local"))
    bundles.deploy_current_tree(None, ["x86-64"])
    assert calls["built"] == 1


def test_deploy_builds_and_ships_only_the_archs_asked_for(monkeypatch):
    """A bundle is built for the machines that will run it, not every arch the
    repo could target: each extra arch costs minutes of building."""
    calls = _deploy_harness(monkeypatch, latest=None)
    manifest = bundles.deploy_current_tree(None, ["znver4", "znver3", "znver4"])
    assert calls["archs"] == [["znver3", "znver4"]]
    assert manifest.archs == ["znver3", "znver4"]


def test_deploy_writes_when_the_store_lacks_an_arch(monkeypatch):
    """LATEST carrying this tree is not enough: it must carry it for every
    arch asked for, or a znver4 machine would find nothing to run."""
    calls = _deploy_harness(monkeypatch, latest=_manifest("local"))  # x86-64 only
    assert bundles.deploy_current_tree(None, ["x86-64"]).bundle_id == "old"
    assert bundles.deploy_current_tree(None, ["znver4"]).bundle_id == "new"
    assert calls["written"] == 1


def _stored(store, bundle_id: str, positions: str, age: int):
    """A bundle in the store, its manifest `age` seconds old."""
    d = store / bundle_id
    d.mkdir(parents=True)
    manifest = bundles.BundleManifest(
        bundle_id=bundle_id, git_sha="s", git_dirty=False, archs=["znver3"],
        eval_positions=positions,
    )  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest.__dict__))
    (d / bundles.arch_tarball_name("znver3")).write_bytes(b"tar")
    os.utime(d / "manifest.json", (1000 + age, 1000 + age))
    bundles.eval_positions_path(store, positions).parent.mkdir(exist_ok=True)
    bundles.eval_positions_path(store, positions).write_bytes(b"pos")


def test_the_store_reads_back_its_bundles(tmp_path):
    _stored(tmp_path, "b1", "p1", 0)
    assert bundles.latest_manifest(tmp_path) is None
    (tmp_path / bundles.LATEST_NAME).write_text("b1\n")
    assert bundles.latest_manifest(tmp_path).eval_positions == "p1"
    assert bundles.arch_tarball(tmp_path, "b1", "znver3").read_bytes() == b"tar"
    assert bundles.read_manifest(tmp_path, "nope") is None


def test_pruning_keeps_pinned_and_recent_bundles_and_their_datasets(tmp_path, monkeypatch):
    """A task pinned to an old bundle must still find it to create a
    container; the rest go, beyond the newest few, and so does any eval
    dataset no remaining bundle names."""
    monkeypatch.setattr(bundles, "KEEP_NEWEST", 2)
    for i in range(5):
        _stored(tmp_path, f"b{i}", f"p{i}", i)
    bundles.prune(tmp_path, keep={"b0"})
    assert sorted(d.name for d in tmp_path.iterdir() if d.name.startswith("b")) == [
        "b0",
        "b3",
        "b4",
    ]
    assert sorted(p.name for p in (tmp_path / bundles.DEPS_DIR).iterdir()) == [
        "positions-p0.tar.gz",
        "positions-p3.tar.gz",
        "positions-p4.tar.gz",
    ]


def test_a_written_bundle_lands_whole_and_leaves_no_work_dir(tmp_path, monkeypatch):
    """The bundle's directory and its datasets' tarball appear by rename, LATEST
    names the bundle, and writing the same bundle again changes nothing."""
    manifest = bundles.BundleManifest(
        bundle_id="b1", git_sha="s", git_dirty=False, archs=["znver3"], eval_positions="p1"
    )

    def create_bundle(out_dir, archs):
        (out_dir / bundles.arch_tarball_name("znver3")).write_bytes(b"tar")
        (out_dir / "manifest.json").write_text(json.dumps(manifest.__dict__))
        return [], manifest

    def create_positions(out_dir):
        (out_dir / "positions.tar.gz").write_bytes(b"pos")
        return out_dir / "positions.tar.gz"

    monkeypatch.setattr(bundles, "create_bundle", create_bundle)
    monkeypatch.setattr(bundles, "create_eval_positions_tarball", create_positions)
    for _ in range(2):
        assert bundles.write_bundle(tmp_path, ["znver3"]) == manifest
        assert sorted(p.name for p in tmp_path.iterdir()) == ["LATEST", "b1", "deps"]
    assert sorted(p.name for p in (tmp_path / "b1").iterdir()) == [
        bundles.arch_tarball_name("znver3"),
        "manifest.json",
    ]
    assert bundles.latest_manifest(tmp_path) == manifest
    assert bundles.eval_positions_path(tmp_path, "p1").read_bytes() == b"pos"
    assert [p.name for p in (tmp_path / "deps").iterdir()] == ["positions-p1.tar.gz"]
