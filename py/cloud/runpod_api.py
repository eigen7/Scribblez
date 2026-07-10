"""Minimal client for the Runpod REST API (https://rest.runpod.io/v1).

Covers exactly what the fleet tooling needs -- creating, listing, and
terminating pods -- over stdlib urllib. Auth is a Bearer API key
(credentials.json: runpod.api_key).
"""

import json
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"

# The public GraphQL endpoint. Unlike the REST API it needs no API key for the
# read-only discovery queries below, but it rejects requests without a
# User-Agent header. It is the only Runpod surface that exposes instance
# catalog, live pricing, and stock; the REST API (RunpodClient) has no
# discovery endpoints.
GRAPHQL_URL = "https://api.runpod.io/graphql"
_USER_AGENT = "scribblez-dashboard"


class RunpodError(Exception):
    """An HTTP or transport failure talking to the Runpod API."""


def _graphql(query: str) -> dict:
    """POST a query to the unauthenticated GraphQL endpoint and return its
    `data` object. Raises RunpodError on transport, HTTP, or GraphQL errors."""
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace").strip()
        raise RunpodError(f"GraphQL -> HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RunpodError(f"GraphQL -> {e.reason}") from e
    if payload.get("errors"):
        raise RunpodError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


_CPU_FLAVORS_QUERY = """
query {
  cpuFlavors {
    id groupName displayName minVcpu maxVcpu ramMultiplier diskLimitPerVcpu
    specifics { stockStatus securePrice }
  }
}
"""

_GPU_TYPES_QUERY = """
query {
  gpuTypes {
    id displayName memoryInGb secureCloud communityCloud
    securePrice communityPrice secureSpotPrice communitySpotPrice maxGpuCount
    lowestPrice(input: {gpuCount: 1}) { stockStatus minVcpu minMemory }
  }
}
"""


def _cpu_offers() -> list[dict]:
    """The six fixed CPU flavors with live on-demand price ($/vCPU/hr) and
    stock. `securePrice` is the on-demand per-vCPU rate; RAM per vCPU is
    `ramMultiplier` GB. CPU spot pricing is not exposed by the API."""
    out = []
    for f in _graphql(_CPU_FLAVORS_QUERY)["cpuFlavors"]:
        spec = f.get("specifics") or {}
        out.append(
            {
                "id": f["id"],
                "display_name": f["displayName"],
                "group_name": f["groupName"],
                "min_vcpu": f["minVcpu"],
                "max_vcpu": f["maxVcpu"],
                "ram_multiplier": f["ramMultiplier"],
                "disk_per_vcpu": f["diskLimitPerVcpu"],
                "price_per_vcpu_hr": spec.get("securePrice") or None,
                "stock": spec.get("stockStatus"),
            }
        )
    return out


def _gpu_offers() -> list[dict]:
    """Available GPU types with per-cloud on-demand and spot pricing. A price
    is reported only for a cloud the type is actually offered in (the raw
    per-cloud price fields carry stale values otherwise). Types with no stock
    or no price in either cloud are dropped."""
    out = []
    for g in _graphql(_GPU_TYPES_QUERY)["gpuTypes"]:
        low = g.get("lowestPrice") or {}
        secure_ok = bool(g.get("secureCloud"))
        community_ok = bool(g.get("communityCloud"))
        secure_price = (g.get("securePrice") or None) if secure_ok else None
        community_price = (g.get("communityPrice") or None) if community_ok else None
        if low.get("stockStatus") is None or (secure_price is None and community_price is None):
            continue
        out.append(
            {
                "id": g["id"],
                "display_name": g["displayName"],
                "vram_gb": g["memoryInGb"],
                "secure_price": secure_price,
                "community_price": community_price,
                "secure_spot_price": (g.get("secureSpotPrice") or None) if secure_ok else None,
                "community_spot_price": (
                    (g.get("communitySpotPrice") or None) if community_ok else None
                ),
                "max_gpu_count": g["maxGpuCount"],
                "stock": low.get("stockStatus"),
                "min_vcpu": low.get("minVcpu"),
                "min_memory_gb": low.get("minMemory"),
                "secure_available": secure_ok,
                "community_available": community_ok,
            }
        )
    return out


def fetch_cloud_offers() -> dict:
    """The live cloud instance catalog for the dashboard's add-worker form:
    `{"cpu": [...], "gpu": [...]}` with the fields the frontend renders and
    pod creation needs. Raises RunpodError on failure."""
    return {"cpu": _cpu_offers(), "gpu": _gpu_offers()}


class RunpodClient:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def _request(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace").strip()
            raise RunpodError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RunpodError(f"{method} {path} -> {e.reason}") from e
        return json.loads(payload) if payload else None

    def list_pods(self) -> list[dict]:
        return self._request("GET", "/pods")

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def create_pod(self, spec: dict) -> dict:
        return self._request("POST", "/pods", spec)

    def delete_pod(self, pod_id: str):
        self._request("DELETE", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str):
        """Stop without terminating; billing drops to disk-only and start_pod
        can resume it."""
        self._request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str):
        """Start or resume a stopped pod (including one Runpod reclaimed from
        an interruptible rental, capacity permitting)."""
        self._request("POST", f"/pods/{pod_id}/start")
