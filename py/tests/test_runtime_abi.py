"""Tests for the dev-image/worker-image runtime ABI check (cloud.runtime_abi)."""

import subprocess

from cloud import runtime_abi

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


def test_the_record_round_trips(tmp_path):
    runtime_abi.write_record(tmp_path, "repo/worker", DEV)
    record = runtime_abi.read_record(tmp_path)
    assert record == {"image": "repo/worker", "versions": DEV}


def test_no_record_is_no_claim(tmp_path):
    assert runtime_abi.read_record(tmp_path) is None


def test_the_image_probe_agrees_with_reading_the_filesystem():
    """The host asks an image what it provides by running a shell snippet in
    it; the dev container reads its own filesystem. The two must agree, or the
    comparison between them is meaningless."""
    res = subprocess.run(runtime_abi.probe_command(), capture_output=True, text=True, check=True)
    assert runtime_abi.parse_versions(res.stdout) == {
        name: version for name, version in runtime_abi.local_versions().items() if version
    }
