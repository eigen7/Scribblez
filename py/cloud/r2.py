"""Helpers for talking to the R2 results bucket via rclone.

rclone is configured entirely through environment variables (no config file):
the `r2:` remote is an S3-compatible endpoint pointed at the operator's
Cloudflare R2 account. The same helpers serve the laptop-side tools (which get
their R2Credentials from the credentials file) and the worker entrypoint
(which gets them from pod environment variables).
"""

import os
import subprocess

from cloud.credentials import R2Credentials

# rclone remote name used in every r2 path, e.g. "r2:<bucket>/kill_test/...".
RCLONE_REMOTE = "r2"


def rclone_env(r2: R2Credentials) -> dict[str, str]:
    """Environment variables defining the `r2:` rclone remote."""
    prefix = f"RCLONE_CONFIG_{RCLONE_REMOTE.upper()}"
    return {
        f"{prefix}_TYPE": "s3",
        f"{prefix}_PROVIDER": "Cloudflare",
        f"{prefix}_ACCESS_KEY_ID": r2.access_key_id,
        f"{prefix}_SECRET_ACCESS_KEY": r2.secret_access_key,
        f"{prefix}_ENDPOINT": r2.endpoint,
        # Uploads otherwise begin with a CreateBucket-if-missing check, which
        # bucket-scoped R2 tokens are not permitted to make (AccessDenied).
        # The bucket always exists; skip the check.
        f"{prefix}_NO_CHECK_BUCKET": "true",
        # The remote is defined entirely by the variables above, but rclone
        # still prints a NOTICE about the config file it did not find -- on
        # every invocation, which for a multi-file upload reads like something
        # going wrong. Point it at an empty config it can find instead.
        "RCLONE_CONFIG": "/dev/null",
    }


def bucket_path(r2: R2Credentials, *parts: str) -> str:
    """An rclone path inside the bucket, e.g. bucket_path(r2, "kill_test", tag)."""
    return "/".join([f"{RCLONE_REMOTE}:{r2.bucket}", *parts])


def rclone(
    r2: R2Credentials,
    *args: str,
    capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run `rclone <args>` with the r2 remote configured."""
    return subprocess.run(
        ["rclone", *args],
        env=os.environ | rclone_env(r2),
        capture_output=capture,
        input=input_text,
        text=True,
    )
