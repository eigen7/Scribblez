"""Loader for the operator's cloud credentials file.

Every laptop-side cloud tool (fleet control, binary pushes, result syncing)
reads its secrets from a single JSON file in the mount directory:

    /workspace/mount/cloud/credentials.json

The file is filled in by hand (scripts/cloud_check_credentials.py writes a
placeholder template when it is absent, and validates a filled-in one against
the live services). It never leaves the machine: workers on rented pods
receive only the narrow subset they need -- the R2 object credentials -- via
pod environment variables at launch.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CREDENTIALS_PATH = Path("/workspace/mount/cloud/credentials.json")

# Value a human still needs to replace. Any field left equal to this (or
# missing entirely) is reported by load_credentials as unfilled.
PLACEHOLDER = "FILL_ME"

TEMPLATE = {
    "runpod": {
        # Runpod console -> Settings -> API Keys.
        "api_key": PLACEHOLDER,
        # Runpod console -> Settings -> Container Registry Auth; the ID of the
        # entry holding the Docker Hub credentials for the worker image.
        "container_registry_auth_id": PLACEHOLDER,
    },
    "registry": {
        # Private worker-image repo, e.g. "docker.io/someuser/scribblez-worker".
        "worker_image": PLACEHOLDER,
    },
    "r2": {
        # Cloudflare dashboard -> R2; the account ID appears in the bucket
        # endpoint URL.
        "account_id": PLACEHOLDER,
        # An R2 API token scoped to the bucket with Object Read & Write.
        "access_key_id": PLACEHOLDER,
        "secret_access_key": PLACEHOLDER,
        "bucket": "scribblez",
    },
}


class CredentialsError(Exception):
    """The credentials file is missing, malformed, or not fully filled in."""


@dataclass(frozen=True)
class RunpodCredentials:
    api_key: str
    container_registry_auth_id: str


@dataclass(frozen=True)
class RegistryConfig:
    worker_image: str


@dataclass(frozen=True)
class R2Credentials:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


@dataclass(frozen=True)
class CloudCredentials:
    runpod: RunpodCredentials
    registry: RegistryConfig
    r2: R2Credentials


def write_template(path: Path = CREDENTIALS_PATH):
    """Write the placeholder template to `path` (must not already exist) with
    owner-only permissions, ready for the operator to fill in."""
    assert not path.exists(), f"{path} already exists; refusing to overwrite"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TEMPLATE, indent=2) + "\n")
    path.chmod(0o600)


def _collect(raw: dict, section: str, field: str, problems: list[str]) -> str:
    value = raw.get(section, {}).get(field)
    if not isinstance(value, str) or not value or value == PLACEHOLDER:
        problems.append(f"{section}.{field}")
        return ""
    return value


def load_credentials(path: Path = CREDENTIALS_PATH) -> CloudCredentials:
    """Parse `path` into CloudCredentials.

    Raises CredentialsError naming every missing or still-placeholder field,
    or if the file itself is absent/unparseable.
    """
    if not path.is_file():
        raise CredentialsError(
            f"No credentials file at {path}. "
            "Run scripts/cloud_check_credentials.py to create a template."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CredentialsError(f"{path} is not valid JSON: {e}") from e

    problems: list[str] = []
    creds = CloudCredentials(
        runpod=RunpodCredentials(
            api_key=_collect(raw, "runpod", "api_key", problems),
            container_registry_auth_id=_collect(
                raw, "runpod", "container_registry_auth_id", problems
            ),
        ),
        registry=RegistryConfig(
            worker_image=_collect(raw, "registry", "worker_image", problems),
        ),
        r2=R2Credentials(
            account_id=_collect(raw, "r2", "account_id", problems),
            access_key_id=_collect(raw, "r2", "access_key_id", problems),
            secret_access_key=_collect(raw, "r2", "secret_access_key", problems),
            bucket=_collect(raw, "r2", "bucket", problems),
        ),
    )
    if problems:
        raise CredentialsError(
            f"Unfilled field(s) in {path}: {', '.join(problems)}. "
            "Edit the file and fill in each value."
        )
    return creds
