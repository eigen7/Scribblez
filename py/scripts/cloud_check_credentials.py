#!/usr/bin/env python3
"""Validate /workspace/mount/cloud/credentials.json against the live services.

If the file is absent, writes a placeholder template and exits. Otherwise
checks each credential by exercising it:

  - R2: writes, reads back, and deletes a probe object in the bucket.
  - Worker image name: shape check only (a private repo's existence can't be
    probed without pulling; the first image push exercises it).
  - Registry pull token: asks Docker Hub's token service for pull access to
    the worker image repo with it.
  - AWS: the access key's identity, and a listing of the instances it can see.

Exits nonzero if any check fails.

Usage:
    ./py/scripts/cloud_check_credentials.py
"""

import base64
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
from cloud.providers.aws import AwsProvider
from cloud.providers.base import ProviderError
from cloud.r2 import bucket_path, rclone

PROBE_OBJECT = "_credentials_check/probe.txt"
PROBE_CONTENT = "scribblez credentials probe\n"


def report(ok: bool, what: str, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}{f': {detail}' if detail else ''}")
    return ok


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


def check_registry_pull_token(creds: CloudCredentials) -> bool:
    """Docker Hub's token service grants a pull scope on the repo to a
    credential that can pull it -- the same exchange `docker pull` makes."""
    repo = creds.registry.worker_image.removeprefix("docker.io/").split(":")[0]
    basic = base64.b64encode(
        f"{creds.registry.username}:{creds.registry.pull_token}".encode()
    ).decode()
    req = urllib.request.Request(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull",
        headers={"Authorization": f"Basic {basic}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return report(False, "registry.pull_token", f"token service -> HTTP {e.code}")
    except urllib.error.URLError as e:
        return report(False, "registry.pull_token", str(e.reason))
    if not body.get("token"):
        return report(False, "registry.pull_token", "no token granted")
    return report(True, "registry.pull_token", f"pull access to {repo}")


def check_aws(creds: CloudCredentials) -> bool:
    provider = AwsProvider(creds.aws, creds.registry)
    try:
        arn = provider.identity()
        instances = provider.describe()
    except ProviderError as e:
        return report(False, "aws", f"{e}: {e.detail}")
    return report(True, "aws", f"{arn} in {creds.aws.region}, {len(instances)} instance(s) ours")


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
        check_r2_round_trip(creds),
        check_worker_image_name(creds),
        check_registry_pull_token(creds),
        check_aws(creds),
    ]
    if all(results):
        print("All checks passed.")
        return 0
    print("Some checks failed; fix the corresponding entries and rerun.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
