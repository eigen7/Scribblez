"""Tests for the dev-image/worker-image runtime ABI check (cloud.runtime_abi)."""

import subprocess

from cloud import runtime_abi
from scribblez.paths import REPO_ROOT

DEV = {
    "libstdc++.so.6": "libstdc++.so.6.0.35",
    "libgcc_s.so.1": "libgcc_s.so.1",
    "libnvinfer.so.10": "libnvinfer.so.10.11.0",
    "libcudart.so.12": "libcudart.so.12.9.79",
}


def test_a_matching_image_is_not_stale():
    assert runtime_abi.stale_libraries(dict(DEV), DEV) == []


def test_an_older_libstdcxx_is_stale():
    """The failure this exists for: gcc-16 in the dev image, stock Ubuntu in
    the worker image, every bundle unloadable."""
    worker = DEV | {"libstdc++.so.6": "libstdc++.so.6.0.33"}
    assert runtime_abi.stale_libraries(worker, DEV) == ["libstdc++.so.6"]


def test_a_newer_libstdcxx_is_fine():
    """It is backward compatible: an image ahead of the dev container still
    runs its bundles."""
    worker = DEV | {"libstdc++.so.6": "libstdc++.so.6.0.36"}
    assert runtime_abi.stale_libraries(worker, DEV) == []


def test_a_mismatched_nvidia_runtime_is_stale_either_way():
    """TensorRT is copied verbatim and version-locked to what the engine was
    built against, so newer is as wrong as older."""
    for version in ("libnvinfer.so.10.9.0", "libnvinfer.so.10.12.0"):
        worker = DEV | {"libnvinfer.so.10": version}
        assert runtime_abi.stale_libraries(worker, DEV) == ["libnvinfer.so.10"]


def test_a_library_the_dev_container_lacks_is_not_judged():
    """Nothing was built against it, so there is nothing to be stale about."""
    dev = DEV | {"libcudart.so.12": ""}
    worker = DEV | {"libcudart.so.12": "libcudart.so.12.0.0"}
    assert runtime_abi.stale_libraries(worker, dev) == []


def test_the_record_round_trips_per_runtime(tmp_path):
    """One record per image: a push of one runtime's image keeps what the
    other's push recorded."""
    runtime_abi.write_record(tmp_path, "engine", "repo/worker", DEV, 2)
    torch_versions = DEV | {"libstdc++.so.6": "libstdc++.so.6.0.36"}
    runtime_abi.write_record(tmp_path, "torch", "repo/worker:latest-torch", torch_versions, 2)
    assert runtime_abi.read_records(tmp_path) == {
        "engine": {"image": "repo/worker", "versions": DEV, "bootstrap": 2},
        "torch": {"image": "repo/worker:latest-torch", "versions": torch_versions, "bootstrap": 2},
    }
    runtime_abi.write_record(tmp_path, "engine", "repo/worker", torch_versions, 2)
    assert runtime_abi.read_records(tmp_path)["engine"]["versions"] == torch_versions


def test_no_record_is_no_claim(tmp_path):
    assert runtime_abi.read_records(tmp_path) is None
    # A record from before images were keyed by runtime says nothing either.
    runtime_abi.record_path(tmp_path).parent.mkdir(parents=True)
    runtime_abi.record_path(tmp_path).write_text('{"image": "w", "versions": {}}')
    assert runtime_abi.read_records(tmp_path) is None


def test_the_image_probe_agrees_with_reading_the_filesystem():
    """The host asks an image what it provides by running a shell snippet in
    it; the dev container reads its own filesystem. The two must agree, or the
    comparison between them is meaningless."""
    res = subprocess.run(runtime_abi.probe_command(), capture_output=True, text=True, check=True)
    assert runtime_abi.parse_versions(res.stdout) == {
        name: version for name, version in runtime_abi.local_versions().items() if version
    }


def test_the_dashboard_and_the_baked_bootstrap_speak_one_protocol():
    """runtime_abi's copy of the protocol and bootstrap.py's must move
    together, or an image built from this tree would be refused, or accepted
    while unable to find the bundle it is handed."""
    text = (REPO_ROOT / "docker-setup" / "worker" / "bootstrap.py").read_text()
    assert f"BOOTSTRAP_PROTOCOL = {runtime_abi.BOOTSTRAP_PROTOCOL}\n" in text
    assert f'PAYLOAD_DIR = Path("{runtime_abi.PAYLOAD_DIR}")\n' in text
