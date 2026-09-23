"""rclone access to the R2 results bucket (Cloudflare's S3-compatible store).

The `r2:` remote is defined entirely through environment variables, with no
rclone config file, so the same helpers work for controller-side tools (whose
R2Credentials come from the credentials file) and for workers (whose come
from the container environment; see sinks.r2_from_env).
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
        # Without this, uploads start with a create-bucket-if-missing call,
        # which a bucket-scoped R2 token may not make (AccessDenied). The
        # bucket always exists.
        f"{prefix}_NO_CHECK_BUCKET": "true",
        # Silences the NOTICE rclone otherwise prints on every invocation
        # about the config file it did not find.
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
