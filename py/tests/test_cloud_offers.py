"""Tests for parsing Runpod GraphQL discovery responses into cloud offers.

The GraphQL endpoint is mocked with canned responses, so these exercise the
field mapping and the GPU availability/pricing filtering without a network call.
"""

import pytest
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


def test_rest_requests_carry_a_user_agent(monkeypatch):
    """Cloudflare in front of the REST API refuses urllib's default signature
    with a 403; without this header no pod can be created or listed."""
    import io
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        seen["auth"] = req.get_header("Authorization")
        return io.BytesIO(b"[]")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert runpod_api.RunpodClient("k").list_pods() == []
    assert seen == {"ua": runpod_api._USER_AGENT, "auth": "Bearer k"}


def test_rest_errors_carry_the_apis_own_message(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    def fail(req, timeout=None):
        body = io.BytesIO(b'{"error":"create pod: no longer any instances","status":500}')
        raise urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, body)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(
        runpod_api.RunpodError,
        match=r"POST /pods -> HTTP 500: create pod: no longer any instances$",
    ):
        runpod_api.RunpodClient("k").create_pod({})


def test_pod_listing_takes_its_runtime_from_graphql(monkeypatch):
    """The REST listing says what a pod should be doing, not whether its
    container is up; that comes from GraphQL, as the account."""
    import io
    import json
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None):
        if req.full_url == runpod_api.GRAPHQL_URL:
            seen["auth"] = req.get_header("Authorization")
            seen["query"] = json.loads(req.data)["query"]
            body = {"data": {"myself": {"pods": [
                {"id": "up", "runtime": {"uptimeInSeconds": 42}},
                {"id": "down", "runtime": None},
            ]}}}  # fmt: skip
        else:
            body = [
                {"id": "up", "desiredStatus": "RUNNING"},
                {"id": "down", "desiredStatus": "EXITED"},
                {"id": "unlisted", "desiredStatus": "RUNNING"},
            ]
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pods = {p["id"]: p for p in runpod_api.RunpodClient("k").list_pods()}
    assert pods["up"]["runtime"] == {"uptimeInSeconds": 42}
    assert pods["down"]["runtime"] is None
    assert pods["unlisted"]["runtime"] is None
    assert seen["auth"] == "Bearer k" and "runtime" in seen["query"]
