"""Tests for parsing Runpod GraphQL discovery responses into cloud offers.

The GraphQL endpoint is mocked with canned responses, so these exercise the
field mapping and the GPU availability/pricing filtering without a network call.
"""

from cloud import runpod_api

_CPU_FLAVORS = [
    {
        "id": "cpu3c", "groupName": "CPU3", "displayName": "Compute-Optimized",
        "minVcpu": 2, "maxVcpu": 32, "ramMultiplier": 2, "diskLimitPerVcpu": 10,
        "specifics": {"stockStatus": "High", "securePrice": 0.06},
    },
    {
        "id": "cpu5m", "groupName": "CPU5", "displayName": "Memory-Optimized",
        "minVcpu": 2, "maxVcpu": 32, "ramMultiplier": 8, "diskLimitPerVcpu": 15,
        "specifics": {"stockStatus": None, "securePrice": 0.09},
    },
]  # fmt: skip

_GPU_TYPES = [
    # Offered in both clouds: on-demand and spot kept for both.
    {
        "id": "NVIDIA A100 80GB PCIe", "displayName": "A100 PCIe", "memoryInGb": 80,
        "secureCloud": True, "communityCloud": True, "securePrice": 1.39, "communityPrice": 1.19,
        "secureSpotPrice": 0.9, "communitySpotPrice": 0.8, "maxGpuCount": 8,
        "lowestPrice": {"stockStatus": "Low", "minVcpu": 12, "minMemory": 117},
    },
    # Community-only: the stale secure price/spot are dropped, community kept.
    {
        "id": "NVIDIA GeForce RTX 3070", "displayName": "RTX 3070", "memoryInGb": 8,
        "secureCloud": False, "communityCloud": True, "securePrice": 5.0, "communityPrice": 0.13,
        "secureSpotPrice": 5.0, "communitySpotPrice": 0.13, "maxGpuCount": 8,
        "lowestPrice": {"stockStatus": "Low", "minVcpu": 8, "minMemory": 17},
    },
    # No stock -> dropped.
    {
        "id": "NVIDIA H100", "displayName": "H100", "memoryInGb": 80,
        "secureCloud": True, "communityCloud": False, "securePrice": 2.0, "communityPrice": 0,
        "secureSpotPrice": None, "communitySpotPrice": None, "maxGpuCount": 8,
        "lowestPrice": {"stockStatus": None, "minVcpu": 16, "minMemory": 100},
    },
    # Priced but offered in neither cloud -> dropped.
    {
        "id": "NVIDIA Retired", "displayName": "Retired", "memoryInGb": 24,
        "secureCloud": False, "communityCloud": False, "securePrice": 1.0, "communityPrice": 1.0,
        "secureSpotPrice": 1.0, "communitySpotPrice": 1.0, "maxGpuCount": 1,
        "lowestPrice": {"stockStatus": "High", "minVcpu": 2, "minMemory": 8},
    },
]  # fmt: skip


def _fake_graphql(cpu_flavors, gpu_types):
    def graphql(query: str) -> dict:
        return {"cpuFlavors": cpu_flavors} if "cpuFlavors" in query else {"gpuTypes": gpu_types}

    return graphql


def test_cpu_offers_map_fields(monkeypatch):
    monkeypatch.setattr(runpod_api, "_graphql", _fake_graphql(_CPU_FLAVORS, []))
    cpu = runpod_api._cpu_offers()
    assert [f["id"] for f in cpu] == ["cpu3c", "cpu5m"]
    c = cpu[0]
    assert c["price_per_vcpu_hr"] == 0.06
    assert c["ram_multiplier"] == 2
    assert (c["min_vcpu"], c["max_vcpu"]) == (2, 32)
    assert c["disk_per_vcpu"] == 10
    assert c["stock"] == "High"
    assert cpu[1]["stock"] is None  # out-of-stock flavor still listed


def test_gpu_offers_filter_and_gate_pricing(monkeypatch):
    monkeypatch.setattr(runpod_api, "_graphql", _fake_graphql([], _GPU_TYPES))
    gpu = runpod_api._gpu_offers()
    # No-stock and offered-nowhere types are dropped.
    assert [g["id"] for g in gpu] == ["NVIDIA A100 80GB PCIe", "NVIDIA GeForce RTX 3070"]

    a100 = gpu[0]
    assert (a100["secure_price"], a100["community_price"]) == (1.39, 1.19)
    assert (a100["secure_spot_price"], a100["community_spot_price"]) == (0.9, 0.8)
    assert a100["vram_gb"] == 80
    assert a100["secure_available"] and a100["community_available"]

    rtx = gpu[1]
    assert rtx["secure_price"] is None  # not on secure cloud; stale value gated out
    assert rtx["secure_spot_price"] is None
    assert rtx["community_price"] == 0.13
    assert rtx["secure_available"] is False
    assert rtx["community_available"] is True


def test_fetch_cloud_offers_shape(monkeypatch):
    monkeypatch.setattr(runpod_api, "_graphql", _fake_graphql(_CPU_FLAVORS, _GPU_TYPES))
    offers = runpod_api.fetch_cloud_offers()
    assert set(offers) == {"cpu", "gpu"}
    assert len(offers["cpu"]) == 2
    assert len(offers["gpu"]) == 2
