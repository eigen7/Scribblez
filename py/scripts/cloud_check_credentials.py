#!/usr/bin/env python3
"""Validate /workspace/mount/cloud/credentials.json against the live services.

If the file is absent, writes a placeholder template and exits. Otherwise
checks each credential by exercising it:

  - Runpod API key: lists the account's pods.
  - Runpod registry auth ID: fetches that container-registry-auth entry.
  - R2: writes, reads back, and deletes a probe object in the bucket.
  - Worker image name: shape check only (a private repo's existence can't be
    probed without pulling; the first image push exercises it).

Exits nonzero if any check fails.

Usage:
    ./py/scripts/cloud_check_credentials.py
"""

import json
import shutil
import sys
import urllib.error
import urllib.request

from cloud.credentials import (
    CREDENTIALS_PATH,
    CloudCredentials,
    CredentialsError,
    load_credentials,
    write_template,
)
from cloud.r2 import bucket_path, rclone

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
PROBE_OBJECT = "_credentials_check/probe.txt"
PROBE_CONTENT = "scribblez credentials probe\n"


def report(ok: bool, what: str, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}{f': {detail}' if detail else ''}")
    return ok


def runpod_get(api_key: str, path: str):
    """GET a Runpod REST endpoint, returning the parsed JSON body."""
    req = urllib.request.Request(
        f"{RUNPOD_API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def check_runpod_api_key(creds: CloudCredentials) -> bool:
    try:
        pods = runpod_get(creds.runpod.api_key, "/pods")
    except urllib.error.HTTPError as e:
        return report(False, "runpod.api_key", f"GET /pods -> HTTP {e.code}")
    except urllib.error.URLError as e:
        return report(False, "runpod.api_key", f"GET /pods -> {e.reason}")
    return report(True, "runpod.api_key", f"account has {len(pods)} pod(s)")


def check_runpod_registry_auth(creds: CloudCredentials) -> bool:
    auth_id = creds.runpod.container_registry_auth_id
    try:
        auth = runpod_get(creds.runpod.api_key, f"/containerregistryauth/{auth_id}")
    except urllib.error.HTTPError as e:
        return report(
            False,
            "runpod.container_registry_auth_id",
            f"GET /containerregistryauth/{auth_id} -> HTTP {e.code}",
        )
    except urllib.error.URLError as e:
        return report(False, "runpod.container_registry_auth_id", str(e.reason))
    return report(True, "runpod.container_registry_auth_id", f"entry '{auth.get('name')}'")


def check_r2_round_trip(creds: CloudCredentials) -> bool:
    if shutil.which("rclone") is None:
        return report(False, "r2", "rclone not installed in this container")
    probe = bucket_path(creds.r2, PROBE_OBJECT)

    write = rclone(creds.r2, "rcat", probe, capture=True, input_text=PROBE_CONTENT)
    if write.returncode != 0:
        return report(False, "r2 write", write.stderr.strip().splitlines()[-1])
    read = rclone(creds.r2, "cat", probe, capture=True)
    if read.returncode != 0 or read.stdout != PROBE_CONTENT:
        return report(False, "r2 read-back", read.stderr.strip().splitlines()[-1])
    delete = rclone(creds.r2, "deletefile", probe, capture=True)
    if delete.returncode != 0:
        return report(False, "r2 delete", delete.stderr.strip().splitlines()[-1])
    return report(True, "r2", f"write/read/delete round-trip in bucket '{creds.r2.bucket}'")


def check_worker_image_name(creds: CloudCredentials) -> bool:
    image = creds.registry.worker_image
    if "/" not in image:
        return report(False, "registry.worker_image", f"'{image}' has no repo path")
    return report(True, "registry.worker_image", image)


def main() -> int:
    if not CREDENTIALS_PATH.is_file():
        write_template()
        print(f"Wrote a placeholder template to {CREDENTIALS_PATH}.")
        print("Fill in each FILL_ME value, then rerun this script.")
        return 1

    try:
        creds = load_credentials()
    except CredentialsError as e:
        print(e)
        return 1

    print(f"Checking credentials from {CREDENTIALS_PATH} ...")
    results = [
        check_runpod_api_key(creds),
        check_runpod_registry_auth(creds),
        check_r2_round_trip(creds),
        check_worker_image_name(creds),
    ]
    if all(results):
        print("All checks passed.")
        return 0
    print("Some checks failed; fix the corresponding entries and rerun.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
