"""AWS EC2 as a machine provider (cloud/providers/base.py).

An instance is launched from the region's current Deep Learning Base GPU AMI
(Ubuntu; the NVIDIA driver, Docker and the container toolkit preinstalled,
found through AWS's public SSM parameter, so there is no image of ours to
build), with a key pair and a security group of ours, tagged as the
dashboard's, and a first-boot script that logs Docker in to the image
registry and pulls both worker images before writing the readiness marker
-- so the pull happens on the machine during `launching`, never on the
controller's blocking thread at a slot's first start.

The catalog is curated: the types that fit the two shapes we run (a GPU
trainer, a CPU generator), with the CPU family that picks the bundle arch
and the on-demand list price beside each. Prices are an estimate's input;
they are changed here by PR when AWS reprices.
"""

import time
from pathlib import Path

import boto3
import botocore.exceptions

from cloud.credentials import AwsCredentials, RegistryConfig
from cloud.providers.base import Instance, LaunchRequest, MachineType, ProviderError

# Where the key pair's private key lives (created by prepare()).
AWS_DIR = Path("/workspace/mount/cloud/aws")
KEY_PAIR_NAME = "scribblez"
SECURITY_GROUP_NAME = "scribblez"
OWNER_TAG = "scribblez"
# The Deep Learning Base GPU AMI's public parameter; AWS keeps it current.
AMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
)
SSH_USER = "ubuntu"
READY_FILE = "/var/lib/scribblez/ready"
# The Deep Learning AMI's root snapshot is 75 GB (2026-09), and a launch that
# asks for less is refused (InvalidBlockDeviceMapping). Above that, room for
# the two worker images, a bundle, and a trainer's generation window.
ROOT_VOLUME_GB = 100

# us-east-1 on-demand list prices, checked 2026-09-15.
CATALOG = [
    MachineType("c7a.xlarge", 4, 0, "", "znver4", 0.205),
    MachineType("c7a.2xlarge", 8, 0, "", "znver4", 0.411),
    MachineType("c7a.4xlarge", 16, 0, "", "znver4", 0.821),
    MachineType("c7a.8xlarge", 32, 0, "", "znver4", 1.642),
    MachineType("g6.2xlarge", 8, 1, "L4 24 GB", "znver3", 0.978),
    MachineType("g6.4xlarge", 16, 1, "L4 24 GB", "znver3", 1.323),
    MachineType("g6.8xlarge", 32, 1, "L4 24 GB", "znver3", 2.014),
]

_STATES = {
    "pending": "pending",
    "running": "running",
    "stopping": "stopping",
    "stopped": "stopped",
    "shutting-down": "terminated",
    "terminated": "terminated",
}

QUOTA_CONSOLE = "https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"


def user_data(registry: RegistryConfig) -> str:
    """The first-boot script. The registry token travels in the instance's
    user data, readable by the instance's own metadata service and by our
    IAM user; it is a read-only pull token, which is the reason it is one.

    cloud-init runs this as root, but the login has to be the ssh user's:
    the dashboard pulls as that user before every container it creates (a
    rebuilt image reaches the machine that way), and Docker credentials are
    per user. The first launch pulled as root and every later pull as
    ubuntu was "access denied" with the images sitting right there."""
    as_user = f"sudo -u {SSH_USER} -H"
    pulls = "\n".join(f"{as_user} docker pull {image}" for image in registry.images)
    return f"""#!/bin/bash
set -e
echo {registry.pull_token} | {as_user} docker login --username {registry.username} --password-stdin
{pulls}
mkdir -p {Path(READY_FILE).parent}
touch {READY_FILE}
"""


def _instance(raw: dict) -> Instance:
    tags = {t["Key"]: t["Value"] for t in raw.get("Tags", [])}
    launched = raw.get("LaunchTime")
    return Instance(
        id=raw["InstanceId"],
        state=_STATES.get(raw["State"]["Name"], raw["State"]["Name"]),
        type_id=raw["InstanceType"],
        owner=tags.get(OWNER_TAG),
        address=raw.get("PublicIpAddress") or None,
        launched_at=launched.timestamp() if launched is not None else None,
    )


def _call(fn, *args, **kwargs):
    """Run one boto3 call, turning its failure into a ProviderError that
    carries AWS's error code and message."""
    try:
        return fn(*args, **kwargs)
    except botocore.exceptions.ClientError as e:
        err = e.response.get("Error", {})
        raise ProviderError(err.get("Code", "error"), err.get("Message", str(e))) from e
    except botocore.exceptions.BotoCoreError as e:
        raise ProviderError(type(e).__name__, str(e)) from e


class AwsProvider:
    name = "aws"
    ssh_user = SSH_USER
    identity_file = str(AWS_DIR / f"{KEY_PAIR_NAME}.pem")
    ready_file = READY_FILE

    def __init__(self, creds: AwsCredentials, registry: RegistryConfig, session=None):
        self.region = creds.region
        self._registry = registry
        session = session or boto3.Session(
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name=creds.region,
        )
        self._ec2 = session.client("ec2")
        self._ssm = session.client("ssm")
        self._quotas = session.client("service-quotas")
        self._sts = session.client("sts")

    def catalog(self) -> list[MachineType]:
        return list(CATALOG)

    # ---- one-time setup ----------------------------------------------------

    def identity(self) -> str:
        return _call(self._sts.get_caller_identity)["Arn"]

    def account(self) -> str:
        """E.g. "AWS account 832300492506 as user scribblez, us-east-1"."""
        arn = self.identity()
        account_id = arn.split(":")[4]
        user = arn.rsplit("/", 1)[-1]
        return f"AWS account {account_id} as user {user}, {self.region}"

    def prepare(self):
        """The account-side setup, idempotent: the key pair (its private key
        saved beside the credentials file), the security group that admits
        ssh, and the AMI lookup. Run by scripts/aws_setup.py and again
        before every launch, so a key or group deleted in the console is
        recreated rather than failing the launch."""
        self._ensure_key_pair()
        self._ensure_security_group()
        self.ami()

    def _ensure_key_pair(self):
        key_path = Path(self.identity_file)
        existing = _call(
            self._ec2.describe_key_pairs, Filters=[{"Name": "key-name", "Values": [KEY_PAIR_NAME]}]
        )
        if existing["KeyPairs"]:
            assert key_path.is_file(), (
                f"the '{KEY_PAIR_NAME}' key pair exists on AWS but its private key is not at "
                f"{key_path}: delete the key pair in the console and rerun, and it is recreated"
            )
            return
        created = _call(self._ec2.create_key_pair, KeyName=KEY_PAIR_NAME, KeyType="ed25519")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(created["KeyMaterial"])
        key_path.chmod(0o600)

    def _security_group_id(self) -> str | None:
        found = _call(
            self._ec2.describe_security_groups,
            Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP_NAME]}],
        )["SecurityGroups"]
        return found[0]["GroupId"] if found else None

    def _ensure_security_group(self) -> str:
        group_id = self._security_group_id()
        if group_id is not None:
            return group_id
        group_id = _call(
            self._ec2.create_security_group,
            GroupName=SECURITY_GROUP_NAME,
            Description="scribblez rented machines: ssh from the controller",
        )["GroupId"]
        _call(
            self._ec2.authorize_security_group_ingress,
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    "Ipv6Ranges": [{"CidrIpv6": "::/0"}],
                }
            ],
        )
        return group_id

    def ami(self) -> str:
        return _call(self._ssm.get_parameter, Name=AMI_PARAMETER)["Parameter"]["Value"]

    def quotas(self) -> dict[str, float]:
        """The vCPU quotas that bound what can be rented, by a short name."""
        codes = {
            "on-demand G/VT (GPU) vCPUs": "L-DB2E81BA",
            "spot G/VT (GPU) vCPUs": "L-3819A6DF",
            "on-demand standard vCPUs": "L-1216C47A",
            "spot standard vCPUs": "L-34B43A08",
        }
        return {
            name: _call(self._quotas.get_service_quota, ServiceCode="ec2", QuotaCode=code)["Quota"][
                "Value"
            ]
            for name, code in codes.items()
        }

    def quota_requests(self) -> list[tuple[str, float, str]]:
        history = _call(self._quotas.list_requested_service_quota_change_history, ServiceCode="ec2")
        return [
            (r["QuotaName"], r["DesiredValue"], r["Status"]) for r in history["RequestedQuotas"]
        ]

    # ---- instances -------------------------------------------------------

    def launch(self, request: LaunchRequest) -> Instance:
        self.prepare()
        raw = _call(
            self._ec2.run_instances,
            ImageId=self.ami(),
            InstanceType=request.type_id,
            KeyName=KEY_PAIR_NAME,
            SecurityGroupIds=[self._security_group_id()],
            MinCount=1,
            MaxCount=1,
            UserData=user_data(self._registry),
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": ROOT_VOLUME_GB,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": [{"Key": OWNER_TAG, "Value": request.owner}]},
                {"ResourceType": "volume", "Tags": [{"Key": OWNER_TAG, "Value": request.owner}]},
            ],
        )["Instances"][0]
        instance = _instance(raw)
        instance.owner = request.owner  # tags are not always echoed on the run response
        instance.launched_at = instance.launched_at or time.time()
        return instance

    def describe(self) -> dict[str, Instance]:
        pages = self._ec2.get_paginator("describe_instances").paginate(
            Filters=[{"Name": "tag-key", "Values": [OWNER_TAG]}]
        )
        out = {}
        for page in _call(list, pages):
            for reservation in page["Reservations"]:
                for raw in reservation["Instances"]:
                    inst = _instance(raw)
                    out[inst.id] = inst
        return out

    def stop(self, instance_id: str):
        _call(self._ec2.stop_instances, InstanceIds=[instance_id])

    def start(self, instance_id: str):
        _call(self._ec2.start_instances, InstanceIds=[instance_id])

    def terminate(self, instance_id: str):
        _call(self._ec2.terminate_instances, InstanceIds=[instance_id])

    def refusal(self, error: ProviderError, type_id: str) -> str:
        """The operator-facing sentence for a launch or start AWS refused:
        what happened and what to do, with AWS's own words after."""
        code = str(error)
        if code == "VcpuLimitExceeded":
            return (
                f"AWS refused a {type_id}: the account's vCPU quota for that instance family "
                f"is used up or still 0. Request an increase at {QUOTA_CONSOLE} (grants take a "
                f"day or two), or pick a smaller type. (AWS: {error.detail})"
            )
        if code in ("InsufficientInstanceCapacity", "Unsupported"):
            return (
                f"No {type_id} capacity in the zone right now. Try again in a few minutes, or "
                f"another type. (AWS: {error.detail})"
            )
        if code in ("UnauthorizedOperation", "AuthFailure", "InvalidClientTokenId"):
            return (
                f"AWS rejected the credentials or the IAM policy for this action: check the "
                f"aws section of the credentials file and the scribblez user's policy. "
                f"(AWS: {error.detail})"
            )
        if code == "PendingVerification":
            return (
                "AWS is still verifying the account; new accounts cannot launch instances "
                f"for up to a day. (AWS: {error.detail})"
            )
        if code == "InvalidBlockDeviceMapping":
            return (
                f"AWS refused the root volume for a {type_id}: the image's snapshot has "
                f"outgrown ROOT_VOLUME_GB in cloud/providers/aws.py. (AWS: {error.detail})"
            )
        if code == "IncorrectInstanceState":
            return (
                "The instance is not in a state that allows this yet; it is retried. "
                f"(AWS: {error.detail})"
            )
        return f"AWS would not do that ({code}): {error.detail}"
