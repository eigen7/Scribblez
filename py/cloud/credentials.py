"""Loader for the operator's cloud credentials file.

Every laptop-side cloud tool (fleet control, binary pushes, result syncing)
reads its secrets from a single JSON file in the mount directory:

    /workspace/mount/cloud/credentials.json

The file is filled in by hand (scripts/cloud_check_credentials.py writes a
placeholder template when it is absent, and validates a filled-in one against
the live services). It never leaves the machine: workers on rented machines
receive only the narrow subset they need -- the R2 object credentials, and a
read-only registry pull token -- through their environment at launch.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_ENGINE, RUNTIMES, TORCH_TAG_SUFFIX

CREDENTIALS_PATH = Path("/workspace/mount/cloud/credentials.json")

# Value a human still needs to replace. Any field left equal to this (or
# missing entirely) is reported by load_credentials as unfilled.
PLACEHOLDER = "FILL_ME"

TEMPLATE = {
    "registry": {
        # Private worker-image repo, e.g. "docker.io/someuser/scribblez-worker".
        "worker_image": PLACEHOLDER,
        # Docker Hub account and a read-only personal access token for it
        # (Account settings -> Personal access tokens): what a rented machine
        # logs in with to pull the worker images.
        "username": PLACEHOLDER,
        "pull_token": PLACEHOLDER,
    },
    "aws": {
        # An IAM user's access key (never the root account's), with the EC2,
        # SSM and Service Quotas actions docs/cloud_machines_plan.md lists,
        # and the region machines are rented in.
        "region": "us-east-1",
        "access_key_id": PLACEHOLDER,
        "secret_access_key": PLACEHOLDER,
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
class RegistryConfig:
    # The engine-runtime worker image, "<repo>[:<tag>]"; the torch-runtime
    # image is the same repo under the tag with TORCH_TAG_SUFFIX appended
    # ("latest" when none is given), so one credential names both.
    worker_image: str
    # A read-only pull credential for that repo, for rented machines.
    username: str = ""
    pull_token: str = ""

    @property
    def images(self) -> list[str]:
        return [self.image_for(runtime) for runtime in RUNTIMES]

    def image_for(self, runtime: str) -> str:
        """The image a role of `runtime` (cloud/runtime_abi.py) runs on."""
        assert runtime in RUNTIMES, runtime
        if runtime == RUNTIME_ENGINE:
            return self.worker_image
        repo, tag = _split_tag(self.worker_image)
        return f"{repo}:{tag}{TORCH_TAG_SUFFIX}"


def _split_tag(image: str) -> tuple[str, str]:
    """("<repo>", "<tag>") of an image name, the tag "latest" when it carries
    none. A colon before the last slash belongs to a registry port."""
    head, _, last = image.rpartition("/")
    repo_last, colon, tag = last.partition(":")
    repo = f"{head}/{repo_last}" if head else repo_last
    return repo, (tag if colon else "latest")


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
class AwsCredentials:
    region: str
    access_key_id: str
    secret_access_key: str


@dataclass(frozen=True)
class CloudCredentials:
    registry: RegistryConfig
    r2: R2Credentials
    aws: AwsCredentials


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
        registry=RegistryConfig(
            worker_image=_collect(raw, "registry", "worker_image", problems),
            username=_collect(raw, "registry", "username", problems),
            pull_token=_collect(raw, "registry", "pull_token", problems),
        ),
        aws=AwsCredentials(
            region=_collect(raw, "aws", "region", problems),
            access_key_id=_collect(raw, "aws", "access_key_id", problems),
            secret_access_key=_collect(raw, "aws", "secret_access_key", problems),
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
