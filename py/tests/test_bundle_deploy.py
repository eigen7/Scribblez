"""Tests for bundle deployment: the source fingerprint and the deploy path
that keeps the bucket's LATEST equal to the controller's tree."""

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
    monkeypatch.setattr(bundles, "_shipped_files", lambda: files)
    first = bundles.source_hash()
    assert first == bundles.source_hash()

    (tmp_path / "b").write_text("three")
    assert bundles.source_hash() != first


def test_source_hash_covers_the_name_a_file_ships_under(tmp_path, monkeypatch):
    """Two archs' binaries can hold identical bytes; swapping which is which
    is still a different bundle."""
    swapped = [(n, p) for n, p in _tree(tmp_path, a="x", b="y")]
    monkeypatch.setattr(bundles, "_shipped_files", lambda: swapped)
    straight = bundles.source_hash()
    monkeypatch.setattr(bundles, "_shipped_files", lambda: [(n, p) for n, p in reversed(swapped)])
    assert bundles.source_hash() != straight


def test_source_hash_is_unknown_while_an_arch_is_unbuilt(tmp_path, monkeypatch):
    files = _tree(tmp_path, a="one") + [("missing", tmp_path / "nope")]
    monkeypatch.setattr(bundles, "_shipped_files", lambda: files)
    assert bundles.source_hash() is None


def test_source_hash_cache_follows_the_file(tmp_path, monkeypatch):
    """The cache exists so a status poll costs a stat walk; it must not hand
    back a digest for content that has since changed."""
    files = _tree(tmp_path, a="one")
    monkeypatch.setattr(bundles, "_shipped_files", lambda: files)
    cache = {}
    before = bundles.source_hash(cache)
    (tmp_path / "a").write_text("changed")
    assert bundles.source_hash(cache) != before


def _manifest(source_hash: str, bundle_id: str = "old") -> bundles.BundleManifest:
    return bundles.BundleManifest(
        bundle_id=bundle_id, git_sha="s", git_dirty=False, archs=["x86-64"], source_hash=source_hash
    )


def _deploy_harness(monkeypatch, *, latest, local_hash="local"):
    calls = {"built": 0, "pushed": 0}

    def build(jobs=None):
        calls["built"] += 1

    def push(r2):
        calls["pushed"] += 1
        return _manifest(local_hash, "new")

    monkeypatch.setattr(bundles, "build_all_supported_archs", build)
    monkeypatch.setattr(bundles, "source_hash", lambda cache=None: local_hash)
    monkeypatch.setattr(bundles, "latest_manifest", lambda r2: latest)
    monkeypatch.setattr(bundles, "push_bundle", push)
    return calls


def test_deploy_pushes_when_the_bucket_is_behind(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=_manifest("stale"))
    assert bundles.deploy_current_tree(None).bundle_id == "new"
    assert (calls["built"], calls["pushed"]) == (1, 1)


def test_deploy_skips_the_push_when_the_bucket_already_has_this_tree(monkeypatch):
    """An id that changed on every deploy would unpin every task that shares
    it, replacing containers to run identical code."""
    calls = _deploy_harness(monkeypatch, latest=_manifest("local"))
    assert bundles.deploy_current_tree(None).bundle_id == "old"
    assert (calls["built"], calls["pushed"]) == (1, 0)


def test_deploy_pushes_when_nothing_has_ever_been_pushed(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=None)
    assert bundles.deploy_current_tree(None).bundle_id == "new"
    assert calls["pushed"] == 1


def test_deploy_pushes_over_a_manifest_that_predates_source_hashes(monkeypatch):
    calls = _deploy_harness(monkeypatch, latest=_manifest(""))
    assert bundles.deploy_current_tree(None).bundle_id == "new"
    assert calls["pushed"] == 1


def test_deploy_always_builds_before_deciding(monkeypatch):
    """The fingerprint covers compiled binaries: deciding without building
    would ship an unbuilt arch under a current-looking id."""
    calls = _deploy_harness(monkeypatch, latest=_manifest("local"))
    bundles.deploy_current_tree(None)
    assert calls["built"] == 1
