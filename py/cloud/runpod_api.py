"""Minimal client for the Runpod REST API (https://rest.runpod.io/v1).

Covers exactly what the fleet tooling needs -- creating, listing, and
terminating pods -- over stdlib urllib. Auth is a Bearer API key
(credentials.json: runpod.api_key).
"""

import json
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"


class RunpodError(Exception):
    """An HTTP or transport failure talking to the Runpod API."""


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
